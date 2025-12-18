"""
Template API Router
템플릿 생성 및 관리 API 엔드포인트
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Union

# 모듈 임포트
from app.agents.agent1 import Agent1
from app.agents.agent2 import Agent2
from app.core.template_selector import TemplateSelector
from app.utils.llm_provider_manager import get_llm_manager
from app.dto.api_result import ApiResult, ErrorResponse
from app.utils.variable_cleaner import clean_variables_list, clean_template_content
from app.utils.language_detector import validate_input_language, ValidationError
from app.utils.industry_purpose_mapping import get_category_info
from app.utils.performance_logger import get_performance_logger, TimingContext
import time
import uuid

router = APIRouter()


class TemplateRequest(BaseModel):
    """템플릿 생성 요청 모델"""

    userId: int
    requestContent: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="구체적인 알림톡 템플릿 요청 내용 (최소 10자 이상)",
        examples=["카페 예약 확인 알림톡 템플릿을 만들어주세요"],
    )
    conversationContext: Optional[str] = Field(
        None, description="재질문 컨텍스트", examples=["이전 대화 내용"]
    )

    @validator("requestContent")
    def validate_request_content(cls, v):
        """requestContent 유효성 검증"""
        if not v or v.strip() == "":
            raise ValueError("템플릿 요청 내용을 입력해주세요")

        # 언어별 검증을 먼저 실행 (영어 입력 감지)
        is_valid, error_type, message = validate_input_language(v)
        if not is_valid:
            if error_type == ValidationError.ENGLISH_ONLY:
                raise ValueError(
                    "Please enter in Korean. English-only input cannot generate KakaoTalk templates."
                )
            else:
                raise ValueError(message)

        # 한국어 기본값이나 의미없는 텍스트 필터링 (영어는 위에서 이미 처리됨)
        invalid_inputs = ["샘플", "테스트", "없음", "기본값"]
        if v.strip().lower() in invalid_inputs:
            raise ValueError("구체적인 템플릿 요청 내용을 입력해주세요")

        # 최소 단어 수 검증 (2단어 이상) - 언어 검증 통과 후에만 체크
        words = v.strip().split()
        if len(words) < 2:
            raise ValueError("더 구체적인 요청 내용을 입력해주세요 (최소 2단어 이상)")

        return v.strip()


class ErrorDetail(BaseModel):
    """에러 상세 정보 모델"""

    code: str
    message: str
    details: Optional[str] = None


class ErrorResponse(BaseModel):
    """에러 응답 모델"""

    error: ErrorDetail
    timestamp: str


class Variable(BaseModel):
    """변수 정보 모델"""

    id: int
    variableKey: str = Field(..., alias="variableKey")
    placeholder: str
    inputType: str = Field(..., alias="inputType")


class IndustryPurposeItem(BaseModel):
    """Industry/Purpose 아이템 모델 (ID + 이름)"""

    id: int
    name: str


class TemplateSuccessData(BaseModel):
    """Java AiTemplateResponse와 호환되는 데이터 구조"""

    id: Optional[int]  # 부분 완성 시 null
    userId: int
    categoryId: str
    title: str
    content: str
    imageUrl: Optional[str] = None
    type: str
    isPublic: Optional[bool] = None
    status: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    buttons: List[dict] = []
    variables: List[Variable]
    industries: List[IndustryPurposeItem] = []  # [{"id": 1, "name": "학원"}]
    purposes: List[IndustryPurposeItem] = []  # [{"id": 2, "name": "공지/안내"}]
    _mapped_variables: Dict[str, str] = {}  # FastAPI 전용 필드


class TemplateSuccessResponse(BaseModel):
    """ApiResult로 래핑된 템플릿 응답 모델"""

    data: Optional[TemplateSuccessData] = None
    message: Optional[str] = None
    error: Optional[ErrorResponse] = None


class IncompleteInfoDetails(BaseModel):
    """추가 정보 필요 상세 모델"""

    confirmed_variables: Optional[Dict[str, str]] = None
    missing_variables: Optional[List[str]] = None
    contextual_question: Optional[str] = None
    original_input: Optional[str] = None
    validation_status: Optional[str] = None
    reasoning: Optional[str] = None
    mapped_variables: Optional[Dict[str, str]] = None
    partial_template: Optional[str] = None
    mapping_coverage: Optional[float] = None
    industry: Optional[List[dict]] = None
    purpose: Optional[List[dict]] = None


class ErrorResponseWithDetails(BaseModel):
    """상세 에러 응답 모델"""

    detail: Dict[str, Any]


# PartialTemplateResponse 클래스 삭제 - 202 응답 제거


def create_error_response(
    error_code: str, message: str, details: Any = None, status_code: int = 400
) -> JSONResponse:
    """Java 호환 에러 응답 생성"""
    api_error_response = ErrorResponse(
        code=error_code, message=message, details=details if details else None
    )

    error_result = ApiResult(data=None, message=None, error=api_error_response)
    return JSONResponse(status_code=status_code, content=error_result.dict())


def determine_template_type(buttons: List[dict] = None) -> str:
    """
    버튼 존재 여부에 따라 템플릿 타입 결정

    Args:
        buttons: 버튼 리스트

    Returns:
        "LINK" if buttons exist, else "MESSAGE"
    """
    if buttons and len(buttons) > 0:
        return "LINK"
    return "MESSAGE"


def convert_industry_purpose_data(
    industry_list: List[dict] = None, purpose_list: List[dict] = None
) -> Dict[str, List]:
    """
    Industry/Purpose 데이터를 ID+이름 객체 배열 형식으로 변환

    Args:
        industry_list: Agent2에서 받은 industry 데이터 [{"id": 1, "name": "학원", "confidence": 0.9}, ...]
        purpose_list: Agent2에서 받은 purpose 데이터 [{"id": 2, "name": "공지/안내", "confidence": 0.8}, ...]

    Returns:
        {
            "industries": [{"id": 1, "name": "학원"}],
            "purposes": [{"id": 2, "name": "공지/안내"}]
        }
    """
    result = {"industries": [], "purposes": []}

    # Industry 처리
    for item in industry_list or []:
        if isinstance(item, dict):
            if "id" in item and "name" in item:
                # Agent2에서 오는 정상적인 형식
                result["industries"].append({"id": item["id"], "name": item["name"]})
            elif "name" in item:
                # name만 있는 경우
                result["industries"].append({"id": 9, "name": item["name"]})  # 기타 ID
            else:
                # 기타 dict 형식
                name = str(item.get("name", item))
                result["industries"].append({"id": 9, "name": name})
        else:
            # 문자열인 경우
            name = str(item)
            result["industries"].append({"id": 9, "name": name})

    # Purpose 처리
    for item in purpose_list or []:
        if isinstance(item, dict):
            if "id" in item and "name" in item:
                # Agent2에서 오는 정상적인 형식
                result["purposes"].append({"id": item["id"], "name": item["name"]})
            elif "name" in item:
                # name만 있는 경우
                result["purposes"].append({"id": 11, "name": item["name"]})  # 기타 ID
            else:
                # 기타 dict 형식
                name = str(item.get("name", item))
                result["purposes"].append({"id": 11, "name": name})
        else:
            # 문자열인 경우
            name = str(item)
            result["purposes"].append({"id": 11, "name": name})

    return result


# create_partial_response 함수 삭제 - 202 응답 제거


def format_existing_template_response(existing_template: Dict[str, Any], user_id: int):
    """
    기존 템플릿을 Java 백엔드 호환 구조로 포맷팅
    """
    from datetime import datetime
    import pytz

    # 기존 템플릿 변수를 Java VariableDto 형식으로 변환
    variables_list = existing_template.get("variables", [])

    # 변수 정리 적용
    cleaned_variables = clean_variables_list(variables_list)
    formatted_variables = []

    for i, var in enumerate(cleaned_variables):
        formatted_variables.append(
            {
                "id": i + 1,
                "variableKey": var.get("variable_key"),
                "placeholder": var.get("placeholder"),
                "inputType": var.get("input_type", "TEXT"),
            }
        )

    # 기본 industry/purpose (기존 템플릿이므로 추론)
    korea_tz = pytz.timezone("Asia/Seoul")
    current_time = (
        datetime.now(korea_tz).replace(tzinfo=None).isoformat(timespec="seconds")
    )

    template_data = TemplateSuccessData(
        id=None,  # AI에서 생성하므로 null
        userId=user_id,
        categoryId="004001",  # 기본 카테고리
        title="기존 템플릿 재사용",
        content=existing_template.get("template", ""),
        imageUrl=None,
        type="MESSAGE",
        isPublic=False,
        status="CREATED",
        createdAt=current_time,
        updatedAt=current_time,
        buttons=[],
        variables=formatted_variables,
        industries=[{"id": 9, "name": "기타"}],
        purposes=[{"id": 1, "name": "공지/안내"}],
    )

    return ApiResult.ok(template_data)


@router.post(
    "/templates",
    tags=["Template Generation"],
    responses={
        200: {
            "model": TemplateSuccessResponse,
            "description": "템플릿 생성 완료",
            "content": {
                "application/json": {
                    "example": {
                        "data": {
                            "id": 1,
                            "userId": 1,
                            "categoryId": "004001",
                            "title": "알림톡",
                            "content": "안녕하세요, #{고객명}님. 독서모임 다음 일정 안내드립니다.\\n\\n▶ 모임명 : 독서모임\\n▶ 일시 : 매주 토요일 오후 2시\\n▶ 장소 : 강남역 스터디카페",
                            "imageUrl": None,
                            "type": "MESSAGE",
                            "buttons": [],
                            "variables": [
                                {
                                    "id": 1,
                                    "variableKey": "고객명",
                                    "placeholder": "#{고객명}",
                                    "inputType": "TEXT",
                                }
                            ],
                        },
                        "message": None,
                        "error": None,
                    }
                }
            },
        },
        400: {
            "model": ErrorResponseWithDetails,
            "description": "잘못된 요청",
            "content": {
                "application/json": {
                    "examples": {
                        "inappropriate_request": {
                            "summary": "부적절한 요청",
                            "value": {
                                "data": None,
                                "message": None,
                                "error": {
                                    "code": "INAPPROPRIATE_REQUEST",
                                    "message": "비즈니스 알림톡에 적합하지 않은 요청입니다.",
                                    "timestamp": "2025-09-29T10:30:00Z",
                                },
                            },
                        },
                        "language_validation": {
                            "summary": "언어 검증 오류",
                            "value": {
                                "data": None,
                                "message": None,
                                "error": {
                                    "code": "LANGUAGE_VALIDATION_ERROR",
                                    "message": "Please enter in Korean. English-only input cannot generate KakaoTalk templates.",
                                    "timestamp": "2025-09-29T10:30:00Z",
                                },
                            },
                        },
                        "profanity_detected": {
                            "summary": "비속어 검출",
                            "value": {
                                "data": None,
                                "message": None,
                                "error": {
                                    "code": "PROFANITY_DETECTED",
                                    "message": "부적절한 언어가 감지되었습니다.",
                                    "timestamp": "2025-09-29T10:30:00Z",
                                },
                            },
                        },
                    }
                }
            },
        },
        408: {
            "model": ErrorResponseWithDetails,
            "description": "요청 시간 초과",
            "content": {
                "application/json": {
                    "example": {
                        "data": None,
                        "message": None,
                        "error": {
                            "code": "PROCESSING_TIMEOUT",
                            "message": "템플릿 생성 시간이 초과되었습니다. 다시 시도해주세요.",
                            "timestamp": "2025-09-29T10:30:00Z",
                        },
                    }
                }
            },
        },
        422: {
            "model": ErrorResponseWithDetails,
            "description": "입력 검증 실패 또는 템플릿 생성 불가",
            "content": {
                "application/json": {
                    "example": {
                        "data": None,
                        "message": None,
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "필수 입력 항목이 누락되었습니다.",
                            "timestamp": "2025-09-29T10:30:00Z",
                        },
                    }
                }
            },
        },
        429: {
            "model": ErrorResponseWithDetails,
            "description": "API 할당량 초과",
            "content": {
                "application/json": {
                    "example": {
                        "data": None,
                        "message": None,
                        "error": {
                            "code": "API_QUOTA_EXCEEDED",
                            "message": "API 할당량을 초과했습니다. 잠시 후 다시 시도해주세요.",
                            "timestamp": "2025-09-29T10:30:00Z",
                        },
                    }
                }
            },
        },
        500: {
            "model": ErrorResponseWithDetails,
            "description": "서버 내부 오류",
            "content": {
                "application/json": {
                    "example": {
                        "data": None,
                        "message": None,
                        "error": {
                            "code": "INTERNAL_SERVER_ERROR",
                            "message": "서버 내부 오류가 발생했습니다.",
                            "timestamp": "2025-09-29T10:30:00Z",
                        },
                    }
                }
            },
        },
    },
)
async def create_template(request: TemplateRequest):
    """
    AI 기반 알림톡 템플릿 생성

    Args:
        request: 템플릿 생성 요청 (userId, requestContent)

    Returns:
        생성된 템플릿 정보 또는 에러 응답
    """
    try:
        # 성능 로깅 초기화
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        perf_logger = get_performance_logger()
        stage_times = {}

        print(
            f"🚀 [REQUEST START] {request_id} - User: {request.userId} - Content: '{request.requestContent[:50]}...'"
        )

        # Agent1 분석 시작
        # 1. Agent1 초기화 및 분석
        with TimingContext(perf_logger, "Agent1_Initialization", request_id) as ctx:
            agent1 = Agent1()
        stage_times["agent1_init"] = ctx.duration

        # 2. 사용자 입력 처리
        with TimingContext(perf_logger, "Agent1_Processing", request_id) as ctx:
            agent1_result = await agent1.process_query_async(
                request.requestContent, conversation_context=request.conversationContext
            )
        stage_times["agent1_processing"] = ctx.duration

        print(f"🔍 [DEBUG] Agent1 처리 완료! 상태: {agent1_result['status']}")

        # 3. Agent1 처리 결과에 따른 분기
        if agent1_result["status"] == "inappropriate_request":
            # 부적절한 요청 검출 (비즈니스 알림톡에 적합하지 않음)
            return create_error_response(
                "INAPPROPRIATE_REQUEST", agent1_result["message"]
            )

        elif agent1_result["status"] == "profanity_retry":
            # 비속어 검출
            return create_error_response("PROFANITY_DETECTED", agent1_result["message"])

        elif agent1_result["status"] == "policy_violation":
            # 정책 위반
            return create_error_response("POLICY_VIOLATION", agent1_result["message"])

        elif agent1_result["status"] not in ["complete", "success"]:
            # 기타 Agent1 오류
            return create_error_response(
                "AGENT1_ERROR",
                f"Agent1 처리 실패: {agent1_result.get('message', '알 수 없는 오류')}",
            )

        # 디버깅: Agent1 결과 출력
        print(f"🔍 [DEBUG] Agent1 결과: {agent1_result}")

        # 4. 기존 템플릿 검색 (새 로직)
        try:
            with TimingContext(
                perf_logger, "Existing_Template_Search", request_id
            ) as ctx:
                template_selector = TemplateSelector()

                existing_template = await template_selector.find_existing_template(
                    user_input=request.requestContent,
                    variables=agent1_result.get("variables", {}),
                    intent=agent1_result.get("intent", {}),
                    user_id=request.userId,
                )
        except Exception as e:
            print(f"❌ [ERROR] 기존 템플릿 검색 중 오류: {e}")
            return create_error_response(
                "TEMPLATE_SEARCH_ERROR",
                f"템플릿 검색 중 오류가 발생했습니다: {str(e)}",
                status_code=500,
            )
        stage_times["existing_template_search"] = ctx.duration

        # 4-A. 기존 템플릿 발견 시 바로 반환
        if existing_template:
            print(f"✅ [EXISTING TEMPLATE FOUND] {request_id} - 기존 템플릿 사용")
            return format_existing_template_response(existing_template, request.userId)

        # 4-B. 기존 템플릿 없음 - Agent2로 새 템플릿 생성
        print(f"🔄 [NEW TEMPLATE NEEDED] {request_id} - 새 템플릿 생성 시작")

        # 5. Agent2로 최종 템플릿 생성
        with TimingContext(perf_logger, "Agent2_Initialization", request_id) as ctx:
            agent2 = Agent2()
        stage_times["agent2_init"] = ctx.duration

        with TimingContext(
            perf_logger, "Agent2_Template_Generation", request_id
        ) as ctx:
            final_template_result, metadata = (
                await agent2.generate_compliant_template_async(
                    user_input=request.requestContent,
                    agent1_variables=agent1_result.get("variables", {}),
                )
            )
        stage_times["agent2_generation"] = ctx.duration

        if not final_template_result:
            return create_error_response(
                "TEMPLATE_GENERATION_FAILED", "템플릿 생성에 실패했습니다"
            )

        # Check if more variables are needed - 422 에러로 변경 (202 대신)
        if final_template_result.get("status") == "need_more_variables":
            missing_vars = final_template_result.get("missing_variables", [])
            return create_error_response(
                "TEMPLATE_INCOMPLETE",
                "템플릿 생성을 위해 추가 정보가 필요합니다.",
                details={
                    "missing_variables": missing_vars,
                    "partial_template": final_template_result.get("template", ""),
                    "suggestions": ["더 구체적인 정보를 포함해서 다시 요청해주세요"],
                },
                status_code=422,
            )

        # Check if template generation failed
        if not final_template_result.get("success"):
            return create_error_response(
                "TEMPLATE_GENERATION_FAILED", "템플릿 생성에 실패했습니다"
            )

        # 6. 성공 응답 반환 (Java 호환 구조)
        # Variables 변환: variable_key → variableKey (변수 정리 적용)
        variables_list = final_template_result.get("variables", [])

        # 변수 정리 적용
        cleaned_variables = clean_variables_list(variables_list)
        formatted_variables = []

        for i, var in enumerate(cleaned_variables):
            formatted_variables.append(
                {
                    "id": i + 1,
                    "variableKey": var.get("variable_key"),
                    "placeholder": var.get("placeholder"),
                    "inputType": var.get("input_type", "TEXT"),
                }
            )

        # Buttons 변환: AI 형식 → Java Backend 형식
        buttons_list = final_template_result.get("buttons", [])
        formatted_buttons = []
        for i, button in enumerate(buttons_list):
            if isinstance(button, dict):
                # AI 형식에서 Java 형식으로 변환
                formatted_button = {
                    "name": button.get("name", "바로가기"),
                    "linkMo": button.get("url_mobile", button.get("linkMo", "")),
                    "linkPc": button.get("url_pc", button.get("linkPc", "")),
                    "linkAnd": button.get("linkAnd"),
                    "linkIos": button.get("linkIos"),
                    "linkType": (
                        "WL"
                        if button.get("type") == "link"
                        else button.get("linkType", "WL")
                    ),
                    "ordering": i + 1,
                }
                formatted_buttons.append(formatted_button)

        # Industry/Purpose 데이터를 기존 및 새로운 형식 둘 다 생성
        converted_data = convert_industry_purpose_data(
            final_template_result.get("industry", []),
            final_template_result.get("purpose", []),
        )

        # 동적 카테고리 결정
        category_info = get_category_info(
            converted_data["industries"], converted_data["purposes"]
        )

        # TemplateSuccessData에 모든 필수 필드 포함
        from datetime import datetime
        import pytz

        # Java LocalDateTime 호환: 시간대 없는 로컬 시간만 (yyyy-MM-dd'T'HH:mm:ss)
        korea_tz = pytz.timezone("Asia/Seoul")
        current_time = (
            datetime.now(korea_tz).replace(tzinfo=None).isoformat(timespec="seconds")
        )

        template_data = TemplateSuccessData(
            id=None,  # Java 백엔드에서 DB 자동생성 ID 사용
            userId=request.userId,
            categoryId=category_info["categoryId"],
            title=category_info["title"],
            content=clean_template_content(final_template_result.get("template", "")),
            imageUrl=None,
            type=determine_template_type(formatted_buttons),
            isPublic=False,  # 기본값
            status="CREATED",  # 기본값
            createdAt=current_time,
            updatedAt=current_time,
            buttons=formatted_buttons,  # 변환된 버튼 사용
            variables=formatted_variables,  # 변환된 변수 사용
            industries=converted_data["industries"],
            purposes=converted_data["purposes"],
            _mapped_variables={},  # 완성된 템플릿은 빈 객체
        )

        # 최종 성능 로깅
        total_time = time.time() - start_time
        stage_times["total"] = total_time
        stage_times["response_formatting"] = total_time - sum(
            [v for k, v in stage_times.items() if k != "total"]
        )

        # 상세 로깅
        perf_logger.log_request_timing(
            request_id=request_id,
            user_id=request.userId,
            request_content=request.requestContent,
            total_time=total_time,
            stage_times=stage_times,
            metadata={
                "status": "success",
                "template_length": len(str(template_data.content or "")),
                "variables_count": len(template_data.variables or []),
                "has_conversation_context": bool(request.conversationContext),
            },
        )

        print(f"✅ [REQUEST SUCCESS] {request_id} - Total: {total_time:.2f}s")

        # ApiResult로 래핑하여 반환
        return ApiResult.ok(template_data)

    except Exception as e:
        # 에러 성능 로깅
        total_time = time.time() - start_time
        error_message = str(e)

        perf_logger.log_error(
            error_msg=f"템플릿 생성 실패: {error_message}",
            request_id=request_id,
            duration=total_time,
        )

        print(
            f"❌ [REQUEST ERROR] {request_id} - Duration: {total_time:.2f}s - Error: {error_message}"
        )

        # 예외 스택 트레이스 출력
        import traceback
        traceback.print_exc()

        # 특정 오류에 대한 세부 처리
        if "Please enter in Korean" in error_message:
            return create_error_response(
                "LANGUAGE_VALIDATION_ERROR", error_message, None, 400
            )
        elif "quota" in error_message.lower() or "rate limit" in error_message.lower():
            return create_error_response(
                "API_QUOTA_EXCEEDED",
                "API 할당량을 초과했습니다. 잠시 후 다시 시도해주세요.",
                error_message,
                429,
            )
        elif "timeout" in error_message.lower():
            return create_error_response(
                "PROCESSING_TIMEOUT",
                "템플릿 생성 시간이 초과되었습니다. 다시 시도해주세요.",
                "AI 처리 시간 초과 (30초 제한)",
                408,
            )
        else:
            return create_error_response(
                "INTERNAL_SERVER_ERROR",
                "서버 내부 오류가 발생했습니다.",
                error_message,
                500,
            )

    except Exception as e:
        print(f"❌ [FATAL ERROR] create_template 함수에서 예외 발생: {e}")
        import traceback

        traceback.print_exc()
        return create_error_response(
            "UNKNOWN_ERROR",
            f"예상치 못한 오류가 발생했습니다: {str(e)}",
            status_code=500,
        )


@router.get(
    "/templates/test",
    tags=["Template Generation"],
    responses={
        200: {"model": TemplateSuccessResponse},
        400: {"model": ErrorResponseWithDetails},
        500: {"model": ErrorResponseWithDetails},
    },
)
async def test_template_generation() -> Dict[str, Any]:
    """
    템플릿 생성 기능 테스트용 엔드포인트
    """
    test_request = TemplateRequest(
        userId=999, requestContent="독서모임 안내 메시지를 만들어주세요"
    )

    return await create_template(test_request)


class LLMStatusResponse(BaseModel):
    """LLM 상태 응답 모델"""

    status: str
    llm_status: Dict[str, Any]
    timestamp: str


@router.get("/llm/status", tags=["Template Generation"])
async def get_llm_status() -> Dict[str, Any]:
    """
    LLM 제공자 상태 간단 확인 (AI명세서.txt 호환)
    """
    try:
        llm_manager = get_llm_manager()
        llm_status = llm_manager.get_status()

        # AI명세서.txt 형식에 맞는 응답 구조
        result_data = {
            "available_providers": llm_status.get("available_providers", []),
            "primary_provider": llm_status.get("primary_provider", "unknown"),
            "failure_counts": llm_status.get("failure_counts", {}),
            "gemini_configured": llm_status.get("gemini_configured", False),
            "openai_configured": llm_status.get("openai_configured", False),
        }

        return ApiResult.ok(result_data)

    except Exception as e:
        return create_error_response("LLM_STATUS_ERROR", f"LLM 상태 확인 실패: {e}")


@router.get(
    "/templates/llm/status",
    tags=["Template Generation"],
    response_model=LLMStatusResponse,
)
async def get_llm_status_detailed() -> Dict[str, Any]:
    """
    템플릿 생성에 사용되는 LLM 상태 확인 (상세 버전)
    """
    try:
        llm_manager = get_llm_manager()
        return {
            "status": "success",
            "llm_status": llm_manager.get_status(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return create_error_response("LLM_STATUS_ERROR", f"LLM 상태 확인 실패: {e}")
