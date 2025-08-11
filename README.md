## vkr

Перед началом работы необходимо экспортировать токен HuggingFace:
```bash
export HF_TOKEN=your_token
```
Транскрипция аудио в группу
```bash
curl -X POST "http://localhost:8000/audio" \
  -F "file=@./sample.wav" \
  -F "group=mygroup"
```
Ответ на вопрос по группе
```bash
curl -X POST "http://localhost:8000/qa" \
  -H "Content-Type: application/json" \
  -d '{"query":"О чем речь в аудио?", "target":"mygroup"}' \
  "http://localhost:8000/qa"
```
Узнать список групп
```bash
curl -s "http://localhost:8000/groups"
```

Ссылка на веб-интерфейс
```bash
http://localhost:7860
```
