"""Project-wide defaults and column detection candidates."""

DEFAULT_OD_GLOB = "data/raw/od/tpss_bcycl_od_statnhm_202503*.csv"
DEFAULT_STATION_GLOB = "data/raw/stations/*.xls*"
DEFAULT_FREQ = "30min"
DEFAULT_TOP_N = 20
DEFAULT_TRAIN_RATIO = 0.75
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_MODEL_DIR = "models"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_MODEL_PATH = "models/dqn_bike_rebalancing.zip"

CSV_ENCODINGS = ("utf-8-sig", "cp949", "euc-kr")

OD_DATE_CANDIDATES = (
    "기준_날짜",
    "기준날짜",
    "집계_날짜",
    "집계날짜",
    "기준_일자",
    "기준일자",
    "일자",
    "날짜",
    "대여일자",
)

OD_TIME_CANDIDATES = (
    "기준_일시",
    "기준일시",
    "기준_시간값",
    "기준시간값",
    "시간값",
    "기준_시간대",
    "기준시간대",
    "시간대",
    "기준_시간",
    "기준시간",
    "집계_시간",
    "집계시간",
    "집계_기준",
    "집계기준",
    "일시",
    "시간",
)

OD_START_ID_CANDIDATES = (
    "시작_대여소_ID",
    "시작대여소ID",
    "시작_대여소번호",
    "시작대여소번호",
    "출발_대여소_ID",
    "출발대여소ID",
    "출발_대여소번호",
    "출발대여소번호",
)

OD_END_ID_CANDIDATES = (
    "종료_대여소_ID",
    "종료대여소ID",
    "종료_대여소번호",
    "종료대여소번호",
    "도착_대여소_ID",
    "도착대여소ID",
    "도착_대여소번호",
    "도착대여소번호",
)

OD_START_NAME_CANDIDATES = (
    "시작_대여소명",
    "시작대여소명",
    "출발_대여소명",
    "출발대여소명",
    "시작_대여소_이름",
    "출발_대여소_이름",
)

OD_END_NAME_CANDIDATES = (
    "종료_대여소명",
    "종료대여소명",
    "도착_대여소명",
    "도착대여소명",
    "종료_대여소_이름",
    "도착_대여소_이름",
)

OD_COUNT_CANDIDATES = (
    "승객수",
    "전체건수",
    "전체_건수",
    "이용건수",
    "건수",
    "count",
    "cnt",
)

STATION_ID_CANDIDATES = (
    "대여소번호",
    "대여소 번호",
    "대여소ID",
    "대여소_ID",
    "관리번호",
    "대여소 관리번호",
    "station_id",
)

STATION_NAME_CANDIDATES = (
    "대여소명",
    "대여소 이름",
    "보관소명",
    "station_name",
)

STATION_LAT_CANDIDATES = (
    "위도",
    "latitude",
    "lat",
)

STATION_LON_CANDIDATES = (
    "경도",
    "longitude",
    "lng",
    "lon",
)

STATION_CAPACITY_CANDIDATES = (
    "거치대수",
    "거치대 수",
    "보관대수",
    "lcd",
    "qr",
    "capacity",
)
