# from langchain.document_loaders import PyPDFLoader
#
# # loader = PyPDFLoader("/Users/csriniv6/Documents/Srinivasu Chinta/srinivas_ch_-_Manager_Software_Engineering3.pdf")
# #
# # pages = loader.load()
# # print(len(pages))
# #
# # page = pages[0]
# #
# # print(page.page_content[:500])



# from langchain.document_loaders.generic import GenericLoader
# from langchain.document_loaders.parsers import OpenAIWhisperParser
# from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
#
# url = "https://www.youtube.com/watch?v=gqKaVgQxEJ0&t=1156s"
# save_dir = "/Users/csriniv6"
#
# loader = GenericLoader(YoutubeAudioLoader([url], save_dir), OpenAIWhisperParser())
# docs = loader.load()
# docs[0].page_content[:500]


# from langchain.document_loaders import WebBaseLoader
# loader = WebBaseLoader("https://raw.githubusercontent.com/srinivaswavy/NLP/main/VectorMatrix/src/CountVectoization.py")
#
# docs = loader.load()
#
# print(docs)
#
# print(docs[0].page_content[:500])
