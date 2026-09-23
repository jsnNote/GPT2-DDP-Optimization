# GPT-2(RoPE 적용) 기반 병렬 학습 및 최적화 기법 적용


## 1. 프로젝트 개요

### 목적
분산 데이터 병렬(DDP)와 최적화 기법 실습


## 2. 기술 스택 및 적용 기법

### 아키텍처
GPT-2 Medium

### 토크나이저
오픈소스 TokenDagger

### 최적화 기법
Mixed Precision(혼합 정밀도), fused AdamW, Weight Tying, RoPE, Gradient Checkpointing(메모리 확보 목적)


## 3. 데이터

### 데이터셋
GPT-2 학습에 사용한 WebText를 재현한 오픈소스 데이터셋 OpenWebText 사용

### 전처리 방식
Openwebtext의 경우 중복, 품질 필터링이 이미 처리되어 있어서 토크나이저로 토큰화만 처리

## 4. 실습환경
- 파이프라인 검증은 Colab에서 T4(무료)를 이용
- 본 학습은 Runpod이라는 클라우드 환경에서 1epoch 학습
  - Base: A100 SXM(유료) * 1
  - DDP: A100 SXM(유료) * 2


## 5. 일정

### 아키텍처 구현
GPT-2 모델을 직접 구현하고 추가로 필요한 포지셔널 인코딩, 최적화 기법 적용

### 파이프라인 검증
본 학습 전에 Colab 환경에서 파이프라인의 정상 동작하는 지 수천 스텝을 통해서 확인 및 디버깅

### 본학습
클라우드 환경에서 A100 * 1(Base)와 A100 * 2(DDP)를 이용한 1epoch 학습  

### 평가
6번 참고


## 6. 평가 방법

### 최적화 성능
- token/sec
- token/sec/GPU: GPU 1장당 처리량
- speedup 배수(2 GPU 처리량 / 1 GPU 처리량)
- 메모리 사용량
  - torch.cuda.max_memory_allocated() 활용

### 추가 평가
적용 후 loss curve의 수렴 정도 평가
- Baseline과 DDP의 정확성 비교(perplexity/loss)


## 7. 확장
 - 추후 프로젝트인 CUDA를 통해서 직접 구현한 FlashAttention와 미리 구현된 FlashAttention를 프로젝트에 적용해서 최적화 성능을 확인 및 구현 수준을 비교 
 - 학습 수준에서의 최적화에서 나아가 추론 및 서빙에서 사용할 수 있는 최적화 기법인 KV cache와 vLLM 등을 적용
 - 추론 과정에서 Tiktoken같은 BPE는 순차적이어서 병목이 생김-> 더 빠른 토크나이저를 사용    


