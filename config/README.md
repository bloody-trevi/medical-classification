# Model Configuration Directory

.json, .xml, .py 등의 형태로 모델 설정 파일 사전 정의 가능

추후, 모델 변경 혹은 모델 세부 설정에 따른 실험이 필요할 시 업데이트 할 예정

## Example (.json)

```json
vit_nct-crc-he
{
    ...
    "dataset": "NCT-CRC-HE-100K",
    "num_classes": 9,

    "model": "vit_base_patch16_224",
    "learning_rate": 0.0001,
    "loss_function": "CrossEntropyLoss",
    "optimizer": "AdamW",

    "epochs": 150,
    "batch_size": 32,
    "num_workers": 16,
    ...
}
```
