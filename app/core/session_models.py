"""
세션 관리를 위한 데이터 모델 클래스
"""
import secrets
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field


class SessionStatus(Enum):
    """세션 상태"""
    ACTIVE = "active"           # 활성 세션
    COMPLETED = "completed"     # 완료된 세션
    EXPIRED = "expired"         # 만료된 세션
    ERROR = "error"            # 오류 상태


@dataclass
class VariableInfo:
    """템플릿 변수 정보"""
    key: str                    # 변수 키 (예: "고객명")
    placeholder: str            # 플레이스홀더 (예: "#{고객명}")
    variable_type: str = "TEXT" # 변수 타입 (TEXT, DATE, NUMBER, etc.)
    required: bool = True       # 필수 여부
    description: Optional[str] = None  # 변수 설명
    example: Optional[str] = None      # 예시 값
    validation_pattern: Optional[str] = None  # 검증 패턴


@dataclass
class SessionData:
    """사용자 세션 데이터"""
    # 기본 정보
    session_id: str
    user_id: int
    created_at: datetime
    last_updated: datetime
    expires_at: datetime
    status: SessionStatus = SessionStatus.ACTIVE

    # 요청 관련
    original_request: str = ""
    conversation_context: Optional[str] = None

    # 템플릿 관련
    template_content: Optional[str] = None
    template_source: Optional[str] = None  # "matched", "generated", "similar"
    template_variables: Dict[str, VariableInfo] = field(default_factory=dict)

    # 사용자 입력 변수
    user_variables: Dict[str, str] = field(default_factory=dict)
    missing_variables: List[str] = field(default_factory=list)
    completion_percentage: float = 0.0

    # Industry/Purpose 데이터
    industries: List[str] = field(default_factory=list)  # 업종 리스트
    purposes: List[str] = field(default_factory=list)    # 용도 리스트

    # Agent 결과
    agent1_result: Optional[Dict] = None
    agent2_result: Optional[Dict] = None
    validation_result: Optional[Dict] = None

    # 메타데이터
    update_count: int = 0
    error_messages: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """초기화 후 처리"""
        if not self.session_id:
            # 8자리 영숫자 세션 ID 생성
            chars = string.ascii_uppercase + string.digits
            self.session_id = ''.join(secrets.choice(chars) for _ in range(8))

        if not self.created_at:
            self.created_at = datetime.now()

        if not self.last_updated:
            self.last_updated = self.created_at

        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(minutes=30)

    def is_expired(self) -> bool:
        """세션 만료 여부 확인"""
        return datetime.now() > self.expires_at

    def extend_expiration(self, minutes: int = 30):
        """세션 만료 시간 연장"""
        self.expires_at = datetime.now() + timedelta(minutes=minutes)
        self.last_updated = datetime.now()

    def update_completion_percentage(self):
        """완성도 퍼센트 계산 및 업데이트"""
        if not self.template_variables:
            self.completion_percentage = 0.0
            return

        total_vars = len(self.template_variables)
        completed_vars = sum(
            1 for var_key in self.template_variables.keys()
            if var_key in self.user_variables and self.user_variables[var_key].strip()
        )

        self.completion_percentage = (completed_vars / total_vars * 100) if total_vars > 0 else 0.0

        # 누락 변수 업데이트
        self.missing_variables = [
            var_key for var_key in self.template_variables.keys()
            if var_key not in self.user_variables or not self.user_variables[var_key].strip()
        ]

    def add_user_variables(self, variables: Dict[str, str]) -> int:
        """사용자 변수 추가 및 업데이트된 변수 수 반환"""
        updated_count = 0

        for var_key, var_value in variables.items():
            if var_value and var_value.strip():
                old_value = self.user_variables.get(var_key, "")
                if old_value != var_value:
                    self.user_variables[var_key] = var_value.strip()
                    updated_count += 1

        if updated_count > 0:
            self.update_count += 1
            self.last_updated = datetime.now()
            self.update_completion_percentage()
            self.extend_expiration()  # 활동 시 자동 연장

        return updated_count

    def set_template_data(self, template: str, variables: List[Dict], source: str = "generated",
                         industries: Optional[List[str]] = None, purposes: Optional[List[str]] = None):
        """템플릿 데이터 설정"""
        self.template_content = template
        self.template_source = source

        # Industry/Purpose 데이터 설정
        if industries is not None:
            self.industries = industries
        if purposes is not None:
            self.purposes = purposes

        # 변수 정보 설정
        self.template_variables = {}
        for var in variables:
            var_key = var.get('variable_key') or var.get('key')
            if var_key:
                self.template_variables[var_key] = VariableInfo(
                    key=var_key,
                    placeholder=var.get('placeholder', f"#{{{var_key}}}"),
                    variable_type=var.get('input_type', 'TEXT'),
                    required=var.get('required', True),
                    description=var.get('description'),
                    example=var.get('example')
                )

        self.update_completion_percentage()
        self.last_updated = datetime.now()

    def get_progress_summary(self) -> Dict[str, Any]:
        """진행 상황 요약 반환"""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "completion_percentage": round(self.completion_percentage, 1),
            "total_variables": len(self.template_variables),
            "completed_variables": len(self.user_variables),
            "missing_variables": self.missing_variables,
            "update_count": self.update_count,
            "expires_in_minutes": max(0, int((self.expires_at - datetime.now()).total_seconds() / 60)),
            "has_template": bool(self.template_content),
            "template_source": self.template_source,
            "last_updated": self.last_updated.isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (직렬화용)"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status.value,
            "original_request": self.original_request,
            "conversation_context": self.conversation_context,
            "template_content": self.template_content,
            "template_source": self.template_source,
            "template_variables": {
                k: {
                    "key": v.key,
                    "placeholder": v.placeholder,
                    "variable_type": v.variable_type,
                    "required": v.required,
                    "description": v.description,
                    "example": v.example
                } for k, v in self.template_variables.items()
            },
            "user_variables": self.user_variables,
            "missing_variables": self.missing_variables,
            "completion_percentage": self.completion_percentage,
            "industries": self.industries,
            "purposes": self.purposes,
            "update_count": self.update_count,
            "error_messages": self.error_messages,
            "validation_errors": self.validation_errors
        }


@dataclass
class SessionStats:
    """세션 통계"""
    total_sessions: int = 0
    active_sessions: int = 0
    completed_sessions: int = 0
    expired_sessions: int = 0
    total_variables_updated: int = 0
    average_completion_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sessions": self.total_sessions,
            "active_sessions": self.active_sessions,
            "completed_sessions": self.completed_sessions,
            "expired_sessions": self.expired_sessions,
            "total_variables_updated": self.total_variables_updated,
            "average_completion_time_minutes": round(self.average_completion_time, 1)
        }