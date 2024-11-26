# sys.path.append(os.getcwd())
from wiki.passage_retrieval import Retriever
import json

retriever = Retriever({})
# retrive_model = "/home/why/hfmodels/facebook/contriever-msmarco"
retrive_model ="/home/hyy/models/contriever-msmarco"

passages_path = "/home/why/ocean_projects/Fact_counterspeech/enwiki_2020_intro_only/enwiki_2020_intro_only/enwiki_2020_dec_intro_only.jsonl"

# passages_path="/home/hyy/datasets/enwiki-latest-pages-articles/enwiki-latest-pages-articles.jsonl"
# passages_embedding_path="/home/hyy/datasets/enwiki-latest-pages-articles/enwiki_latest_pages_contriever_articles/*"

passages_embedding_path = "/home/why/ocean_projects/Fact_counterspeech/enwiki_2020_intro_only/enwiki_2020_intro_only/enwiki_dec_2020_contriever_intro/*"
retriever.setup_retriever_demo(retrive_model, passages_path, passages_embedding_path,  n_docs=10, save_or_load_index=False)

def search_wiki_doc(querylist,num_docs):
    retriv_docs = []
    for query in querylist:
        retrieved_documents = retriever.search_document_demo(query, num_docs)   # list[]
        retriv_docs.append(retrieved_documents) # list[[]]
    return retriv_docs


# query_3 = ["Who is eligible for council housing in the UK and how does this apply to asylum seekers and refugees?","In which year was the People's Republic of China founded?"]
# query_3 = ["Please retrieve relevant information about Martin Lawrence.",  
# "Martin Fitzgerald Lawrence is an actor, film director, film producer, screenwriter, and comedian.", "Martin Fitzgerald Lawrence is an actor, film director, film producer, screenwriter, and comedian. He came to fame during the 1990s, establishing a Hollywood career as a leading actor, most notably in the films House Party, Bad Boys, Blue Streak, Big Momma's House and A Thin Line Between Love & Hate."]
# Jewish American Council on Philanthropy

# query_3 = ["Ocean pollution global problem",'Jewish American Council on Philanthropy','Martin Fitzgerald Lawrence is an actor, film director, film producer, screenwriter, and comedian.']
# retrive_docs = search_wiki_doc(query_3,10)

# json_retrive_results = json.dumps(retrive_docs, indent=4)
# with open("/home/kayla/lhy/code/multi-source/retrieval_results_sx_4.json", 'w') as json_file:
#     json_file.write(json_retrive_results)




