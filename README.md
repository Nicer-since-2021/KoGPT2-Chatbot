# KoGPT2-Chatbot
자세한 코드 설명은 https://mysterious-world.tistory.com/2 에 있다. 

[emotion_classifications_chatbot_pytorch_kobert+kogpt2.ipynb](https://github.com/Nicer-since-2021/multiclass-emotion-classification-using-KoBERT/blob/main/S00MIN-KIM/emotion_classifications_chatbot_pytorch_kobert%2Bkogpt2.ipynb)에서 다음과 같은 코드에서 사용되는 코드이다. 
해당 코드는 S00MIN-KIM의 레포에서 가져오므로 이 레포에서 불러오려면 조정이 필요하다. (fork한 동일 코드)

```Python
# KoGPT2-chatbot 소스 코드 복사
!git clone --recurse-submodules https://github.com/S00MIN-KIM/KoGPT2-Chatbot.git
```

## 학습 데이터 경로
```Python
# 폴더 이동
%cd /content/drive/MyDrive/kobert/models
```
[emotion_classifications_chatbot_pytorch_kobert+kogpt2.ipynb](https://github.com/Nicer-since-2021/multiclass-emotion-classification-using-KoBERT/blob/main/S00MIN-KIM/emotion_classifications_chatbot_pytorch_kobert%2Bkogpt2.ipynb)에서 위 코드를 통해 폴더 이동을 한 후 파인튜닝을 진행한다. 

그리고 이 [KoGPT2-Chatbot레포의 train_torch](https://github.com/Nicer-since-2021/KoGPT2-Chatbot/blob/main/train_torch.py)의 소스코드를 통해 데이터를 불러와 학습시킨다. 
```Python
def train_dataloader(self):
        data = pd.read_csv('Chatbot_data/ChatbotData.csv', encoding='cp949')
        self.train_set = CharDataset(data, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader
```
따라서 데이터 파일의 최종경로는 다음과 같이 설정되어 있다. 
%cd /content/drive/MyDrive/kobert/models/Chatbot_data_ChatbotData.csv

### KoGPT2 챗봇 학습 데이터
1. [챗봇 트레이닝용 문답 페어](https://github.com/songys/Chatbot_data)

2. [웰니스 대화 스크립트 데이터셋](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-006)
  
  ※ (<챗봇의 응답> | <사용자의 발화>, <감정 라벨>) 형식이 되도록 라벨링 작업이 필요하다.
<img src="https://user-images.githubusercontent.com/68471619/145357192-bf3639a2-a33d-4db0-93c7-3efc9780db4f.png" width="800" height="300"/>

**데이터 파일의 경우 제공받은 데이터를 가공하여 사용했기 때문에 재업로드시 문제가 생길 수 있음을 염려하여 드라이브를 통해 학습했다. 팀 드라이브에 파일 업로드 완료**

이 챗봇 데이터와 코드가 어떻게 활용되어 어떤 결과를 도출하는지 등의 기타 관련 정보는 [다음](https://github.com/Nicer-since-2021/multiclass-emotion-classification-using-KoBERT/tree/main/S00MIN-KIM)을 참고.
