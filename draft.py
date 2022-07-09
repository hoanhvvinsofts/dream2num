# # import phonlp
# # import time
# # import underthesea

# # from underthesea.pipeline.word_tokenize import word_tokenize

# # sent_tk = word_tokenize("Bị một đứa con gái bắt nạt", format="text")

# # model = phonlp.load(save_dir='./pretrained_phonlp')       # 14s

# # a = model.annotate(text=sent_tk)                    # 2s
# # tokens = a[0][0]
# # verbs_type = a[1][0]

# # for verb_type, i in enumerate(verbs_type):
# #     if verb_type != "N" or verb_type != "V":
# #         print(verb_type, tokens[i])

# from googletrans import Translator
# translator = Translator()
# import time

# laos = "ອັນນີ້ລາຄາເທົ່າໃດ?"

# start = time.time()

# eng = translator.translate(str(laos), dest='en').text
# print(eng)

# viet = translator.translate(str(eng), dest='vi').text
# print(viet)

# end = time.time()
# print("time_cost:", end-start)

