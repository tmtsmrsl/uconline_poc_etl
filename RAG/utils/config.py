# Choose between 'zilliz' and 'azure' for VECTOR_SEARCH_DB
VECTOR_SEARCH_DB = "azure" 

# The following variables are used if VECTOR_SEARCH_DB is set to 'zilliz'
SPARSE_EMBEDDINGS_PATH = "artifact/emgt605/sparse_embeddings_v3.joblib"
COLBERT_MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
ZILLIZ_COLLECTION_NAME = "emgt_605_bge_bm25_500_50_v3"

# The following variables are used if VECTOR_SEARCH_DB is set to 'azure'
AZURE_INDEX_NAME = "emgt605_v4"

OUTPUT_FIELDS = [
            "pk",
            "metadata",
            "index_metadata",
            "text"
        ]
COURSE_NAME = "Sustainability Systems in Engineering"
LLM_TEMPERATURE = 0
LLM_MAX_RETRIES = 3
