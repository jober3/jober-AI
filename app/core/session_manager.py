"""
메모리 기반 세션 관리 시스템
Thread-safe 동시 접근 처리 및 자동 만료 관리
"""
import threading
import secrets
import string
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import defaultdict

from .session_models import SessionData, SessionStatus, SessionStats, VariableInfo


logger = logging.getLogger(__name__)


def generate_session_id() -> str:
    """
    8자리 영숫자 세션 ID 생성 (암호학적으로 안전)

    Returns:
        8자리 영숫자 문자열 (예: "A7K9M2X5")
    """
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(8))


class SessionManager:
    """
    메모리 기반 세션 관리자
    - Thread-safe 동시 접근 처리
    - 자동 만료 세션 정리
    - 세션 통계 및 모니터링
    """

    def __init__(self, session_timeout_minutes: int = 30, cleanup_interval_minutes: int = 5):
        """
        세션 관리자 초기화

        Args:
            session_timeout_minutes: 세션 만료 시간 (분)
            cleanup_interval_minutes: 정리 작업 간격 (분)
        """
        # 메모리 저장소 (Thread-Safe)
        self._sessions: Dict[str, SessionData] = {}
        self._user_sessions: Dict[int, List[str]] = defaultdict(list)  # 사용자별 세션 목록
        self._lock = threading.RLock()  # 재귀 락 (데드락 방지)

        # 설정
        self._session_timeout = timedelta(minutes=session_timeout_minutes)
        self._cleanup_interval = cleanup_interval_minutes * 60  # 초 단위

        # 통계
        self._stats = SessionStats()
        self._completion_times: List[float] = []  # 완료 시간 기록 (평균 계산용)

        # 자동 정리 타이머
        self._cleanup_timer: Optional[threading.Timer] = None
        self._start_cleanup_timer()

        logger.info(f"SessionManager 초기화 완료: timeout={session_timeout_minutes}분, cleanup={cleanup_interval_minutes}분")

    def create_session(self, user_id: int, original_request: str,
                      conversation_context: Optional[str] = None) -> str:
        """
        새 세션 생성

        Args:
            user_id: 사용자 ID
            original_request: 원본 요청 텍스트
            conversation_context: 대화 컨텍스트

        Returns:
            생성된 세션 ID
        """
        with self._lock:
            # 중복 방지를 위한 재시도 (최대 5회)
            max_retries = 5
            for attempt in range(max_retries):
                session_id = generate_session_id()
                if session_id not in self._sessions:
                    break
                if attempt == max_retries - 1:
                    logger.warning(f"세션 ID 생성 재시도 {max_retries}회 실패")

            now = datetime.now()

            session = SessionData(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                last_updated=now,
                expires_at=now + self._session_timeout,
                status=SessionStatus.ACTIVE,
                original_request=original_request,
                conversation_context=conversation_context
            )

            # 세션 저장
            self._sessions[session_id] = session
            self._user_sessions[user_id].append(session_id)

            # 통계 업데이트
            self._stats.total_sessions += 1
            self._stats.active_sessions += 1

            logger.info(f"새 세션 생성: {session_id} (사용자: {user_id})")
            return session_id

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        세션 조회

        Args:
            session_id: 세션 ID

        Returns:
            세션 데이터 (없거나 만료된 경우 None)
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            # 만료 확인
            if session.is_expired():
                self._expire_session(session_id)
                return None

            return session

    def get_user_sessions(self, user_id: int, include_expired: bool = False) -> List[SessionData]:
        """
        사용자의 모든 세션 조회

        Args:
            user_id: 사용자 ID
            include_expired: 만료된 세션 포함 여부

        Returns:
            사용자 세션 목록
        """
        with self._lock:
            session_ids = self._user_sessions.get(user_id, [])
            sessions = []

            for session_id in session_ids:
                session = self._sessions.get(session_id)
                if session:
                    if include_expired or not session.is_expired():
                        sessions.append(session)

            return sessions

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        세션 데이터 업데이트

        Args:
            session_id: 세션 ID
            updates: 업데이트할 데이터

        Returns:
            업데이트 성공 여부
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.is_expired():
                return False

            # 허용된 필드만 업데이트
            allowed_fields = {
                'conversation_context', 'template_content', 'template_source',
                'agent1_result', 'agent2_result', 'validation_result'
            }

            updated = False
            for field, value in updates.items():
                if field in allowed_fields and hasattr(session, field):
                    setattr(session, field, value)
                    updated = True

            if updated:
                session.last_updated = datetime.now()
                session.extend_expiration()
                logger.debug(f"세션 업데이트: {session_id}")

            return updated

    def update_user_variables(self, session_id: str, variables: Dict[str, str]) -> bool:
        """
        사용자 변수 업데이트

        Args:
            session_id: 세션 ID
            variables: 업데이트할 변수들

        Returns:
            업데이트 성공 여부
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.is_expired():
                return False

            # 변수 업데이트
            updated_count = session.add_user_variables(variables)

            if updated_count > 0:
                self._stats.total_variables_updated += updated_count
                logger.info(f"세션 {session_id}: {updated_count}개 변수 업데이트 완료")
                return True

            return False

    def set_template_data(self, session_id: str, template: str,
                         variables: List[Dict], source: str = "generated",
                         industries: Optional[List[str]] = None,
                         purposes: Optional[List[str]] = None) -> bool:
        """
        템플릿 데이터 설정

        Args:
            session_id: 세션 ID
            template: 템플릿 내용
            variables: 변수 목록
            source: 템플릿 소스
            industries: 업종 리스트
            purposes: 용도 리스트

        Returns:
            설정 성공 여부
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.is_expired():
                return False

            session.set_template_data(template, variables, source, industries, purposes)
            logger.info(f"세션 {session_id}: 템플릿 데이터 설정 완료 ({len(variables)}개 변수)")
            return True

    def complete_session(self, session_id: str) -> bool:
        """
        세션 완료 처리

        Args:
            session_id: 세션 ID

        Returns:
            완료 처리 성공 여부
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            # 완료 처리
            session.status = SessionStatus.COMPLETED
            session.last_updated = datetime.now()

            # 완료 시간 기록 (통계용)
            completion_time = (session.last_updated - session.created_at).total_seconds() / 60
            self._completion_times.append(completion_time)

            # 최근 100개만 유지 (메모리 절약)
            if len(self._completion_times) > 100:
                self._completion_times = self._completion_times[-100:]

            # 통계 업데이트
            self._stats.active_sessions -= 1
            self._stats.completed_sessions += 1
            if self._completion_times:
                self._stats.average_completion_time = sum(self._completion_times) / len(self._completion_times)

            logger.info(f"세션 완료: {session_id} (소요시간: {completion_time:.1f}분)")
            return True

    def delete_session(self, session_id: str) -> bool:
        """
        세션 삭제

        Args:
            session_id: 세션 ID

        Returns:
            삭제 성공 여부
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if not session:
                return False

            # 사용자 세션 목록에서 제거
            user_id = session.user_id
            if user_id in self._user_sessions:
                try:
                    self._user_sessions[user_id].remove(session_id)
                    if not self._user_sessions[user_id]:  # 빈 리스트면 제거
                        del self._user_sessions[user_id]
                except ValueError:
                    pass

            # 통계 업데이트
            if session.status == SessionStatus.ACTIVE:
                self._stats.active_sessions -= 1

            logger.info(f"세션 삭제: {session_id}")
            return True

    def _expire_session(self, session_id: str):
        """세션 만료 처리 (내부 메서드)"""
        session = self._sessions.get(session_id)
        if session and session.status == SessionStatus.ACTIVE:
            session.status = SessionStatus.EXPIRED
            self._stats.active_sessions -= 1
            self._stats.expired_sessions += 1
            logger.debug(f"세션 만료: {session_id}")

    def cleanup_expired_sessions(self) -> int:
        """
        만료된 세션 정리

        Returns:
            정리된 세션 수
        """
        with self._lock:
            now = datetime.now()
            expired_sessions = []

            # 만료된 세션 찾기
            for session_id, session in self._sessions.items():
                if now > session.expires_at:
                    expired_sessions.append(session_id)

            # 만료된 세션 삭제
            for session_id in expired_sessions:
                self._expire_session(session_id)
                # 완료된 세션은 2시간 후 삭제, 나머지는 즉시 삭제
                session = self._sessions[session_id]
                if session.status != SessionStatus.COMPLETED or now > session.expires_at + timedelta(hours=2):
                    self.delete_session(session_id)

            if expired_sessions:
                logger.info(f"만료된 세션 {len(expired_sessions)}개 정리 완료")

            return len(expired_sessions)

    def _start_cleanup_timer(self):
        """자동 정리 타이머 시작"""
        def cleanup_task():
            try:
                expired_count = self.cleanup_expired_sessions()
                if expired_count > 0:
                    logger.info(f"자동 정리: {expired_count}개 세션 정리")
            except Exception as e:
                logger.error(f"세션 정리 중 오류: {e}")
            finally:
                # 다음 정리 작업 예약
                self._cleanup_timer = threading.Timer(self._cleanup_interval, cleanup_task)
                self._cleanup_timer.daemon = True
                self._cleanup_timer.start()

        # 최초 정리 작업 시작
        self._cleanup_timer = threading.Timer(self._cleanup_interval, cleanup_task)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def get_stats(self) -> SessionStats:
        """세션 통계 조회"""
        with self._lock:
            return SessionStats(
                total_sessions=self._stats.total_sessions,
                active_sessions=self._stats.active_sessions,
                completed_sessions=self._stats.completed_sessions,
                expired_sessions=self._stats.expired_sessions,
                total_variables_updated=self._stats.total_variables_updated,
                average_completion_time=self._stats.average_completion_time
            )

    def get_session_list(self, limit: int = 50, status_filter: Optional[SessionStatus] = None) -> List[Dict[str, Any]]:
        """
        세션 목록 조회 (관리/디버깅용)

        Args:
            limit: 최대 반환 개수
            status_filter: 상태 필터

        Returns:
            세션 요약 정보 목록
        """
        with self._lock:
            sessions = []
            count = 0

            for session in sorted(self._sessions.values(), key=lambda s: s.last_updated, reverse=True):
                if status_filter and session.status != status_filter:
                    continue

                sessions.append(session.get_progress_summary())
                count += 1

                if count >= limit:
                    break

            return sessions

    def shutdown(self):
        """세션 관리자 종료"""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()

        with self._lock:
            active_count = self._stats.active_sessions
            self._sessions.clear()
            self._user_sessions.clear()

        logger.info(f"SessionManager 종료: {active_count}개 활성 세션 정리")


# 전역 세션 관리자 인스턴스
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """싱글톤 세션 관리자 반환"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def create_session_for_user(user_id: int, original_request: str,
                           conversation_context: Optional[str] = None) -> str:
    """사용자 세션 생성 편의 함수"""
    manager = get_session_manager()
    return manager.create_session(user_id, original_request, conversation_context)


def get_user_session(session_id: str) -> Optional[SessionData]:
    """세션 조회 편의 함수"""
    manager = get_session_manager()
    return manager.get_session(session_id)