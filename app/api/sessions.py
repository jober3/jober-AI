"""
Session Management API Router
세션 관리 및 실시간 챗 API 엔드포인트
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field

# 세션 관련 모듈 임포트
from app.core.session_manager import get_session_manager, create_session_for_user, get_user_session
from app.core.template_preview import get_preview_generator
from app.core.session_models import SessionStatus
from app.dto.api_result import ApiResult, ErrorResponse
from app.api.templates import IndustryPurposeItem, convert_industry_purpose_data, determine_template_type
from app.utils.industry_purpose_mapping import get_category_info

router = APIRouter()


# Pydantic 모델 정의
class VariableUpdateRequest(BaseModel):
    """변수 업데이트 요청 모델"""
    variables: Dict[str, str]


class IndustryPurposeUpdateRequest(BaseModel):
    """Industry/Purpose 업데이트 요청 모델"""
    industries: Optional[List[str]] = None
    purposes: Optional[List[str]] = None


class CompleteTemplateRequest(BaseModel):
    """템플릿 완성 요청 모델"""
    final_adjustments: Optional[Dict[str, str]] = None
    force_complete: bool = False


class SessionVariableInfo(BaseModel):
    """세션 변수 정보 모델"""
    variable_key: str = Field(..., alias='variableKey')
    placeholder: str
    variable_type: str = Field(..., alias='variableType')
    required: bool = True
    description: Optional[str] = None
    example: Optional[str] = None
    input_hint: Optional[str] = Field(None, alias='inputHint')
    priority: int = 0


class SessionPreviewResponse(BaseModel):
    """세션 미리보기 응답 모델"""
    success: bool
    session_id: str = Field(..., alias='sessionId')
    preview_template: str = Field(..., alias='previewTemplate')
    completion_percentage: float = Field(..., alias='completionPercentage')
    total_variables: int = Field(..., alias='totalVariables')
    completed_variables: int = Field(..., alias='completedVariables')
    missing_variables: List[SessionVariableInfo] = Field(..., alias='missingVariables')
    next_suggested_variables: List[SessionVariableInfo] = Field(..., alias='nextSuggestedVariables')
    quality_score: float = Field(..., alias='qualityScore')
    estimated_final_length: int = Field(..., alias='estimatedFinalLength')
    readiness_for_completion: bool = Field(..., alias='readinessForCompletion')


class VariableUpdateResponse(BaseModel):
    """변수 업데이트 응답 모델"""
    success: bool
    session_id: str
    updated_variables: List[str]
    completion_percentage: float
    remaining_variables: List[str]
    preview_snippet: str
    quality_score: float
    next_suggested_variables: List[dict]
    session_status: str
    update_count: int
    last_updated: str


class SessionSummary(BaseModel):
    """세션 완료 요약 모델"""
    session_id: str
    total_updates: int
    completion_time_minutes: float
    final_completion_percentage: float
    template_source: str
    quality_score: float


class CompleteTemplateResponse(BaseModel):
    """템플릿 완성 응답 모델"""
    id: int
    userId: int
    categoryId: str
    title: str
    content: str
    imageUrl: Optional[str]
    type: str
    buttons: List[dict]
    variables: List[dict]
    industries: List[IndustryPurposeItem] = []
    purposes: List[IndustryPurposeItem] = []
    session_summary: SessionSummary


class SessionStats(BaseModel):
    """세션 통계 모델"""
    success: bool
    stats: Dict[str, Any]
    timestamp: str


class SessionListResponse(BaseModel):
    """세션 목록 응답 모델"""
    success: bool
    sessions: List[Dict[str, Any]]
    total_count: int
    limit: int
    status_filter: Optional[str]
    timestamp: str


class SessionInfoResponse(BaseModel):
    """세션 정보 응답 모델"""
    success: bool
    session: Dict[str, Any]
    progress_summary: Dict[str, Any]
    timestamp: str


class SessionDeleteResponse(BaseModel):
    """세션 삭제 응답 모델"""
    success: bool
    message: str
    timestamp: str


def create_error_response(error_code: str, message: str, details: Any = None) -> Dict[str, Any]:
    """Java 호환 에러 응답 생성"""
    error_result = ApiResult.error(error_code, message)
    return error_result.dict()


# ===========================================
# 세션 기반 챗봇 API 엔드포인트들
# ===========================================

@router.post("/templates/{session_id}/variables", tags=["Real-time Chat"], response_model=VariableUpdateResponse)
async def update_session_variables(session_id: str, request: VariableUpdateRequest):
    """
    세션의 변수를 개별 업데이트

    Args:
        session_id: 세션 ID
        request: 변수 업데이트 요청

    Returns:
        업데이트 결과 및 현재 세션 상태
    """
    try:
        session_manager = get_session_manager()

        # 세션 유효성 검증
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "SESSION_NOT_FOUND",
                    "세션을 찾을 수 없거나 만료되었습니다.",
                    f"세션 ID: {session_id}"
                )
            )

        # 변수 유효성 검증
        if not request.variables:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "EMPTY_VARIABLES",
                    "업데이트할 변수가 없습니다.",
                    "최소 1개 이상의 변수를 제공해야 합니다."
                )
            )

        # 변수 업데이트
        success = session_manager.update_user_variables(session_id, request.variables)
        if not success:
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    "UPDATE_FAILED",
                    "변수 업데이트에 실패했습니다.",
                    "세션 상태를 확인해주세요."
                )
            )

        # 업데이트된 세션 정보 조회
        updated_session = session_manager.get_session(session_id)

        # 미리보기 생성
        preview_generator = get_preview_generator()
        preview_result = preview_generator.generate_preview(updated_session)

        return {
            "success": True,
            "session_id": session_id,
            "updated_variables": list(request.variables.keys()),
            "completion_percentage": updated_session.completion_percentage,
            "remaining_variables": updated_session.missing_variables,
            "preview_snippet": preview_result.get("preview_template", "")[:100] + "..." if len(preview_result.get("preview_template", "")) > 100 else preview_result.get("preview_template", ""),
            "quality_score": preview_result.get("quality_analysis", {}).get("quality_score", 0),
            "next_suggested_variables": preview_result.get("next_suggested_variables", []),
            "session_status": updated_session.status.value,
            "update_count": updated_session.update_count,
            "last_updated": updated_session.last_updated.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "INTERNAL_ERROR",
                "변수 업데이트 중 오류가 발생했습니다.",
                str(e)
            )
        )


@router.put("/templates/{session_id}/industry-purpose", tags=["Real-time Chat"])
async def update_session_industry_purpose(session_id: str, request: IndustryPurposeUpdateRequest):
    """
    세션의 Industry/Purpose 정보 업데이트

    Args:
        session_id: 세션 ID
        request: Industry/Purpose 업데이트 요청

    Returns:
        업데이트 결과
    """
    try:
        session_manager = get_session_manager()

        # 세션 유효성 검증
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "SESSION_NOT_FOUND",
                    "세션을 찾을 수 없거나 만료되었습니다.",
                    f"세션 ID: {session_id}"
                )
            )

        # Industry/Purpose 업데이트
        if request.industries is not None:
            session.industries = request.industries
        if request.purposes is not None:
            session.purposes = request.purposes

        session.last_updated = datetime.now()

        return {
            "success": True,
            "session_id": session_id,
            "industries": session.industries,
            "purposes": session.purposes,
            "message": "Industry/Purpose 정보가 성공적으로 업데이트되었습니다."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "INTERNAL_ERROR",
                "Industry/Purpose 업데이트 중 오류가 발생했습니다.",
                str(e)
            )
        )


@router.get("/templates/{session_id}/preview", tags=["Real-time Chat"], response_model=SessionPreviewResponse)
async def get_template_preview(session_id: str, style: str = Query("missing", description="미리보기 스타일")):
    """
    부분 완성 템플릿 미리보기 조회

    Args:
        session_id: 세션 ID
        style: 미리보기 스타일 ("missing", "partial", "preview")

    Returns:
        미리보기 템플릿 및 완성도 정보
    """
    try:
        session_manager = get_session_manager()

        # 세션 조회
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "SESSION_NOT_FOUND",
                    "세션을 찾을 수 없거나 만료되었습니다.",
                    f"세션 ID: {session_id}"
                )
            )

        # 템플릿이 설정되어 있는지 확인
        if not session.template_content:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "NO_TEMPLATE",
                    "세션에 템플릿이 설정되지 않았습니다.",
                    "먼저 템플릿을 생성해주세요."
                )
            )

        # 미리보기 생성
        preview_generator = get_preview_generator()
        preview_result = preview_generator.generate_preview(session, style)

        if not preview_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    "PREVIEW_GENERATION_FAILED",
                    "미리보기 생성에 실패했습니다.",
                    preview_result.get("error", "알 수 없는 오류")
                )
            )

        # 응답 데이터 변환
        missing_variables = []
        for var_key, var_info in preview_result.get("missing_variables", {}).items():
            missing_variables.append(SessionVariableInfo(
                variableKey=var_key,
                placeholder=var_info["placeholder"],
                variableType=var_info["variable_type"],
                required=var_info["required"],
                description=var_info.get("description"),
                example=var_info.get("example"),
                inputHint=var_info.get("input_hint"),
                priority=var_info.get("priority", 0)
            ))

        next_suggested = []
        for var_info in preview_result.get("next_suggested_variables", []):
            next_suggested.append(SessionVariableInfo(
                variableKey=var_info["variable_key"],
                placeholder=var_info["placeholder"],
                variableType=var_info["variable_type"],
                required=var_info["required"],
                description=var_info.get("description"),
                example=var_info.get("example"),
                inputHint=var_info.get("input_hint"),
                priority=var_info.get("priority", 0)
            ))

        return SessionPreviewResponse(
            success=True,
            sessionId=session_id,
            previewTemplate=preview_result["preview_template"],
            completionPercentage=round(preview_result["completion_percentage"], 1),
            totalVariables=preview_result["total_variables"],
            completedVariables=preview_result["completed_variables"],
            missingVariables=missing_variables,
            nextSuggestedVariables=next_suggested,
            qualityScore=round(preview_result.get("quality_analysis", {}).get("quality_score", 0), 1),
            estimatedFinalLength=preview_result.get("preview_metadata", {}).get("estimated_final_length", 0),
            readinessForCompletion=preview_result.get("quality_analysis", {}).get("readiness_for_completion", False)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "INTERNAL_ERROR",
                "미리보기 조회 중 오류가 발생했습니다.",
                str(e)
            )
        )


@router.post("/templates/{session_id}/complete", tags=["Real-time Chat"], response_model=CompleteTemplateResponse)
async def complete_template_session(session_id: str, request: CompleteTemplateRequest):
    """
    세션 템플릿 최종 완성

    Args:
        session_id: 세션 ID
        request: 완성 요청

    Returns:
        최종 완성된 템플릿 데이터
    """
    try:
        session_manager = get_session_manager()

        # 세션 조회
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "SESSION_NOT_FOUND",
                    "세션을 찾을 수 없거나 만료되었습니다.",
                    f"세션 ID: {session_id}"
                )
            )

        # 템플릿 존재 확인
        if not session.template_content:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "NO_TEMPLATE",
                    "완성할 템플릿이 없습니다.",
                    "먼저 템플릿을 생성해주세요."
                )
            )

        # 최종 조정사항 적용 (있는 경우)
        if request.final_adjustments:
            session_manager.update_user_variables(session_id, request.final_adjustments)
            session = session_manager.get_session(session_id)  # 업데이트된 세션 다시 조회

        # 완성도 검증 (강제 완성이 아닌 경우)
        if not request.force_complete and session.completion_percentage < 100.0:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "INCOMPLETE_TEMPLATE",
                    "템플릿이 아직 완성되지 않았습니다.",
                    {
                        "completion_percentage": session.completion_percentage,
                        "missing_variables": session.missing_variables,
                        "suggestion": "누락된 변수를 모두 입력하거나 force_complete=true로 강제 완성하세요."
                    }
                )
            )

        # 최종 템플릿 생성
        preview_generator = get_preview_generator()
        final_preview = preview_generator.generate_preview(session, "missing")
        final_template = final_preview["preview_template"]

        # 세션 완료 처리
        session_manager.complete_session(session_id)

        # 변수 목록 변환
        variables_list = []
        for i, (var_key, var_info) in enumerate(session.template_variables.items()):
            variables_list.append({
                "id": i + 1,
                "variableKey": var_key,
                "placeholder": var_info.placeholder,
                "inputType": var_info.variable_type,
                "value": session.user_variables.get(var_key, "")
            })

        # Industry/Purpose 데이터를 세션에서 가져오기
        session_industries = session.industries if session.industries else []
        session_purposes = session.purposes if session.purposes else []

        # Industry/Purpose 데이터 변환 (기존 및 새로운 형식)
        converted_data = convert_industry_purpose_data(session_industries, session_purposes)

        # 동적 카테고리 결정
        category_info = get_category_info(session_industries, session_purposes)

        response_data = {
            "id": None,  # Java 백엔드에서 DB 자동생성 ID 사용
            "userId": session.user_id,
            "categoryId": category_info["categoryId"],
            "title": category_info["title"],
            "content": final_template,
            "imageUrl": None,
            "type": determine_template_type([]),  # 세션에는 현재 버튼정보 없음, 추후 개선 필요
            "buttons": [],
            "variables": variables_list,
            "industries": converted_data["industries"],
            "purposes": converted_data["purposes"]
        }

        # 세션 완료 요약 추가
        response_data["session_summary"] = {
            "session_id": session_id,
            "total_updates": session.update_count,
            "completion_time_minutes": round((session.last_updated - session.created_at).total_seconds() / 60, 1),
            "final_completion_percentage": session.completion_percentage,
            "template_source": session.template_source,
            "quality_score": final_preview.get("quality_analysis", {}).get("quality_score", 0)
        }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "INTERNAL_ERROR",
                "템플릿 완성 중 오류가 발생했습니다.",
                str(e)
            )
        )


# ===========================================
# 세션 관리 및 모니터링 API
# ===========================================

@router.get("/sessions/stats", tags=["Session Management"], response_model=SessionStats)
async def get_session_stats():
    """세션 통계 조회 (관리용)"""
    session_manager = get_session_manager()
    stats = session_manager.get_stats()
    stats_data = {
        "success": True,
        "stats": stats.to_dict(),
        "timestamp": datetime.now().isoformat()
    }
    return ApiResult.ok(stats_data)


@router.get("/sessions", tags=["Session Management"], response_model=SessionListResponse)
async def get_session_list(limit: int = Query(20, description="조회할 세션 수"),
                          status: Optional[str] = Query(None, description="세션 상태 필터")):
    """세션 목록 조회 (관리/디버깅용)"""
    session_manager = get_session_manager()

    # 상태 필터 처리
    status_filter = None
    if status:
        try:
            status_filter = SessionStatus(status.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "INVALID_STATUS",
                    f"유효하지 않은 상태값입니다: {status}",
                    "가능한 값: active, completed, expired, error"
                )
            )

    sessions = session_manager.get_session_list(limit=limit, status_filter=status_filter)

    return {
        "success": True,
        "sessions": sessions,
        "total_count": len(sessions),
        "limit": limit,
        "status_filter": status,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/sessions/{session_id}", tags=["Session Management"], response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    """개별 세션 정보 조회"""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail=create_error_response(
                "SESSION_NOT_FOUND",
                "세션을 찾을 수 없거나 만료되었습니다.",
                f"세션 ID: {session_id}"
            )
        )

    return {
        "success": True,
        "session": session.to_dict(),
        "progress_summary": session.get_progress_summary(),
        "timestamp": datetime.now().isoformat()
    }


@router.delete("/sessions/{session_id}", tags=["Session Management"], response_model=SessionDeleteResponse)
async def delete_session(session_id: str):
    """세션 삭제 (관리용)"""
    session_manager = get_session_manager()
    success = session_manager.delete_session(session_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=create_error_response(
                "SESSION_NOT_FOUND",
                "삭제할 세션을 찾을 수 없습니다.",
                f"세션 ID: {session_id}"
            )
        )

    return {
        "success": True,
        "message": f"세션 {session_id}이 성공적으로 삭제되었습니다.",
        "timestamp": datetime.now().isoformat()
    }