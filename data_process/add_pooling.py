from sentence_transformers import models, SentenceTransformer
from transformers import AutoConfig

base = "matheoqtb/qwen_V2"
cfg = AutoConfig.from_pretrained(base)

word = models.Transformer(base, max_seq_length=8192)
pool = models.Pooling(
    word.get_word_embedding_dimension(),           # = cfg.hidden_size
    pooling_mode="lasttoken"                       # même pooling que Qwen Embedding
)
norm = models.Normalize()

st_model = SentenceTransformer(modules=[word, pool, norm])
st_model.save("./qwen_V2-embedding-lasttoken")

