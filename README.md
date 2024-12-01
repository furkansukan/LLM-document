# Python ile LLM'ler Kullanarak Doküman Analizi 🧠

Bu proje, Python ile büyük dil modelleri (LLM) kullanarak dokümanları analiz etmeyi hedefler. Metin çıkarma ve analiz süreçlerini daha hızlı ve verimli hale getirmek için farklı teknikler kullanılmıştır. Bu rehber, metin özetleme, soru üretimi ve cevaplama işlemlerini adım adım açıklamaktadır.

## 📑 Dokümanı Özetleme

T5-small modelini kullanarak, uzun metinleri daha kısa ve öz hale getiriyorum. Bu, belgenin ana noktalarını hızlıca kavrayabilmenizi sağlar. Uzun belgelerden ana fikri hızla çıkararak zaman kazanmanıza yardımcı olur.

## 🤖 LLM'ler ile Pasajlardan Soru Üretimi

Transformers kütüphanesi ve t5-small modelini kullanarak her pasaj için sorular üretiyorum. Bu işlem, dokümanın önemli noktalarına odaklanmayı sağlar ve analiz için daha verimli bir yol sunar. Bu şekilde, okuyucu metnin kritik noktaları üzerinde derinleşebilir.

## ❓ Üretilen Soruları Bir QA Modeli ile Cevaplama

Son olarak, roberta-base-squad2 modelini kullanarak üretilen soruları, doküman içeriğiyle ilişkilendirip yanıtlıyorum. Bu aşama, metnin doğru ve hızlı bir şekilde anlaşılmasına yardımcı olur. Yanıtlar, kullanıcıların metinle ilgili daha derinlemesine bilgi edinmesini sağlar.

## 🔧 Kullanılan Araçlar ve Kütüphaneler

Bu süreçte aşağıdaki güçlü araçlar ve kütüphaneler kullanılmıştır:
- **pdfplumber**: PDF dosyalarından metin çıkarmak için kullanılır.
- **transformers**: Hugging Face tarafından sağlanan önceden eğitilmiş dil modellerini kullanmak için kullanılır.
- **nltk**: Doğal dil işleme (NLP) için temel kütüphanelerden biridir.
- **roberta-base-squad2**: Soru-cevap modeli, metinlere dayalı sorulara cevap verir.
- **t5-small**: Metin özetleme ve metin tabanlı görevlerde kullanılan bir dil modelidir.

##### **İletişim**
Herhangi bir sorunuz veya geri bildiriminiz için aşağıdaki kanallardan ulaşabilirsiniz:
- **Email:** [furkansukan10@gmail.com](furkansukan10@gmail.com)
- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/furkansukan/)
- **Kaggle:** [Kaggle Profile](https://www.kaggle.com/furkansukan)
