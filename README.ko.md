# TatVTON — Tattoo Virtual Try-On

**SAM 2** 마스크 추출과 **SDXL Inpainting + ControlNet + IP-Adapter** 합성을 이용한 사실적인 타투 합성.

바디 사진, 타투 디자인, 타겟 영역을 입력하면 피부 텍스처, 조명, 신체 윤곽을 보존하면서 타투를 자연스럽게 합성한 결과를 생성합니다.

**[English](./README.md)**

## 아키텍처

### 추론 파이프라인

```mermaid
flowchart TB
    subgraph Input["입력"]
        BODY[바디 이미지]
        TATTOO[타투 디자인]
        REGION[영역 프롬프트<br/>Point / BBox]
    end

    subgraph Phase1["Phase 1 — 마스크 추출"]
        SAM2[SAM 2<br/>Segment Anything 2]
        MASK[피부 마스크]
        BODY --> SAM2
        REGION --> SAM2
        SAM2 --> MASK
    end

    subgraph Phase1b["Phase 1b — 신체 구조 (선택)"]
        DP[DensePose]
        IUV[IUV 맵]
        BODY --> DP
        DP --> IUV
    end

    subgraph Phase2["Phase 2 — 전처리"]
        RESIZE[1024×1024 리사이즈]
        DILATE[마스크 팽창 + 페더링]
        CLIP_EMB[CLIP 임베딩<br/>224×224]
        MASK --> DILATE
        DILATE --> RESIZE
        BODY --> RESIZE
        TATTOO --> CLIP_EMB
    end

    subgraph Phase3["Phase 3 — 디퓨전 인페인팅"]
        SDXL[SDXL Inpainting]
        CN[ControlNet<br/>신체 구조 가이드]
        IPA[IP-Adapter<br/>타투 스타일 가이드]
        RESIZE --> SDXL
        IUV --> CN
        CN --> SDXL
        CLIP_EMB --> IPA
        IPA --> SDXL
        SDXL --> RAW[인페인팅 결과]
    end

    subgraph Phase4["Phase 4 — 합성"]
        COMP[알파 블렌딩<br/>마스크 페더링]
        RAW --> COMP
        RESIZE --> COMP
        DILATE --> COMP
        COMP --> OUTPUT[출력 이미지]
    end

    style Phase1 fill:#e8f4fd,stroke:#1e88e5
    style Phase1b fill:#fce4ec,stroke:#e53935
    style Phase2 fill:#fff3e0,stroke:#fb8c00
    style Phase3 fill:#e8f5e9,stroke:#43a047
    style Phase4 fill:#f3e5f5,stroke:#8e24aa
```

### 모듈 구조

```mermaid
graph LR
    subgraph Entry["진입점"]
        CLI[cli.py]
        API["__init__.py<br/>(Python API)"]
    end

    subgraph Core["코어 파이프라인"]
        PIPE[TatVTONPipeline<br/>pipeline/tatvton_pipeline.py]
        CFG[TatVTONConfig<br/>config.py]
        TYPES[types.py<br/>PointPrompt, BBoxPrompt<br/>MaskResult, PipelineOutput]
    end

    subgraph Models["모델 관리"]
        ML[ModelLoader<br/>models/model_loader.py]
        HUB[Hub 다운로드<br/>models/hub.py]
    end

    subgraph Pre["전처리"]
        SME[SkinMaskExtractor<br/>SAM 2]
        DPE[DensePoseExtractor<br/>detectron2]
        VAL[InputValidator]
    end

    subgraph Engine["추론 엔진"]
        IE[InpaintingEngine<br/>SDXL + ControlNet<br/>+ IP-Adapter]
    end

    subgraph Post["후처리"]
        COMP2[Compositor<br/>알파 블렌딩]
    end

    subgraph Utils["유틸리티"]
        IMG[image.py]
        MSK[mask.py]
        DEV[device.py]
        IMP[imports.py]
    end

    CLI --> PIPE
    API --> PIPE
    CFG --> PIPE
    PIPE --> ML
    ML --> HUB
    PIPE --> SME
    PIPE --> DPE
    PIPE --> VAL
    PIPE --> IE
    PIPE --> COMP2
    IE --> ML
    SME --> ML

    style Entry fill:#e3f2fd
    style Core fill:#e8f5e9
    style Models fill:#fff3e0
    style Pre fill:#fce4ec
    style Engine fill:#f3e5f5
    style Post fill:#e0f7fa
    style Utils fill:#f5f5f5
```

### 학습 파이프라인 (Colab)

```mermaid
flowchart LR
    subgraph Data["데이터 수집"]
        CRAWL[Reddit 크롤러<br/>+ CLIP 필터]
        RAW_IMG[5,400+ 타투 이미지]
        CRAWL --> RAW_IMG
    end

    subgraph Label["라벨링"]
        DPOSE[DensePose]
        IUV_MAP[IUV 맵]
        BODY_LABEL[신체 부위 라벨]
        RAW_IMG --> DPOSE
        DPOSE --> IUV_MAP
        DPOSE --> BODY_LABEL
    end

    subgraph Build["데이터셋 구축"]
        CAPTION[캡션 생성]
        HF_DS[HuggingFace Dataset<br/>image + conditioning + text]
        BODY_LABEL --> CAPTION
        RAW_IMG --> HF_DS
        IUV_MAP --> HF_DS
        CAPTION --> HF_DS
    end

    subgraph Train["ControlNet 학습"]
        SDXL_BASE[SDXL 베이스 모델]
        CN_TRAIN[ControlNet SDXL<br/>파인튜닝]
        CN_MODEL[학습된 ControlNet<br/>가중치]
        SDXL_BASE --> CN_TRAIN
        HF_DS --> CN_TRAIN
        CN_TRAIN --> CN_MODEL
    end

    style Data fill:#e8f4fd,stroke:#1e88e5
    style Label fill:#fce4ec,stroke:#e53935
    style Build fill:#fff3e0,stroke:#fb8c00
    style Train fill:#e8f5e9,stroke:#43a047
```

## 데이터셋

HuggingFace Hub에 큐레이팅된 타투 이미지 데이터셋을 제공합니다:

**[rlaope/tatvton-tattoo-raw](https://huggingface.co/datasets/rlaope/tatvton-tattoo-raw)** — Reddit에서 크롤링한 5,400+ 타투 이미지, CLIP으로 품질 필터링 (피부 위 타투 검증). 6개 신체 부위 x 12가지 스타일.

## 설치

```bash
pip install -e .
```

SAM 2 (별도 설치 필요):

```bash
pip install "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
```

DensePose 지원 (선택):

```bash
pip install -e ".[densepose]"
```

개발 도구:

```bash
pip install -e ".[dev]"
```

## 빠른 시작

### Python API

```python
from PIL import Image
from tatvton import TatVTONPipeline, PointPrompt

body = Image.open("body.jpg")
tattoo = Image.open("tattoo.png")

pipe = TatVTONPipeline()
result = pipe(
    body_image=body,
    tattoo_image=tattoo,
    region=PointPrompt(coords=[(300, 400)]),
)
result.image.save("output.png")
```

### CLI

```bash
# 포인트 프롬프트
tatvton body.jpg tattoo.png --point 300,400 -o output.png

# 바운딩 박스 프롬프트
tatvton body.jpg tattoo.png --bbox 100,150,400,600 -o output.png

# 여러 포인트 + 옵션
tatvton body.jpg tattoo.png --point 300,400 --point 350,420 \
    --steps 20 --strength 0.9 --seed 42 -o output.png

# 마스크와 원본 인페인팅 결과도 저장
tatvton body.jpg tattoo.png --point 300,400 --save-mask --save-raw
```

모듈로도 실행 가능:

```bash
python -m tatvton body.jpg tattoo.png --point 300,400
```

### CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `body` | 바디 이미지 경로 (필수) | - |
| `tattoo` | 타투 디자인 이미지 경로 (필수) | - |
| `--point X,Y` | 포인트 프롬프트 (반복 가능) | - |
| `--bbox X1,Y1,X2,Y2` | 바운딩 박스 프롬프트 | - |
| `-o, --output` | 결과 저장 경로 | `output.png` |
| `--resolution` | 파이프라인 해상도 | 1024 |
| `--steps` | 추론 스텝 수 | 30 |
| `--strength` | 인페인팅 강도 | 0.85 |
| `--guidance-scale` | 가이던스 스케일 | 7.5 |
| `--ip-adapter-scale` | IP-Adapter 스케일 | 0.6 |
| `--seed` | 랜덤 시드 | 랜덤 |
| `--device` | cuda / cpu / mps | cuda |
| `--offload` | none / model / sequential | model |
| `--save-mask` | 마스크 이미지 저장 | - |
| `--save-raw` | 원본 인페인팅 결과 저장 | - |

## 설정

세 가지 수준의 설정:

| 수준 | 방법 | 예시 |
|------|------|------|
| 기본값 | `TatVTONPipeline()` | 12 GB GPU 최적화 |
| Config | `TatVTONPipeline(TatVTONConfig(resolution=768))` | 세션 단위 |
| 호출별 | `pipe(..., strength=0.95, seed=42)` | 호출 단위 |

## 요구 사항

- Python 3.9+
- PyTorch 2.0+
- NVIDIA GPU 12+ GB VRAM (권장)

## 라이선스

MIT
