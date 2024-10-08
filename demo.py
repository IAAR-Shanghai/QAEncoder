import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
model_name = "facebook/contriever"
# model_name = "BAAI/bge-large-en-v1.5"
# model_name = "intfloat/multilingual-e5-large"
model = SentenceTransformer(model_name)

docs = [
    "Who was the astronaut that landed on the moon as part of the Apollo 11 mission?",
    "When did Neil Armstrong land on the moon?",
    "Where did Neil Armstrong land as part of the Apollo 11 mission?",
    "What did Neil Armstrong achieve during the Apollo 11 mission?",
    "Why did Neil Armstrong land on the moon?",
    "How did Neil Armstrong successfully land on the moon?",
    "On July 20, 1969, Neil Armstrong landed on the moon to accomplish the Apollo 11 mission, which was achieved through NASA's extensive planning and the Saturn V rocket.",
]

# docs = [
#     "Who developed the theory of general relativity?",
#     "In what year did Albert Einstein develop the theory of general relativity?",
#     "In which city did Albert Einstein develop the theory of general relativity?",
#     "What scientific theory did Albert Einstein develop in 1915?",
#     "Why did Albert Einstein develop the theory of general relativity?",
#     "How did Albert Einstein explain the force of gravity in his theory of general relativity?",
#     "Albert Einstein, in 1915 in Berlin, developed the theory of general relativity to explain the force of gravity by describing the curvature of spacetime caused by mass and energy.",
# ]

# docs = [
#     "Who won the Nobel Prize in Physics in 1903 for their research on radioactivity?",
#     "When did Marie Curie win the Nobel Prize in Physics?",
#     "Where was the Nobel Prize in Physics awarded to Marie Curie in 1903?",
#     "Why was Marie Curie awarded the Nobel Prize in Physics in 1903?",
#     "What prestigious award did Marie Curie receive in 1903?",
#     "How did Marie Curie conduct her research that led to winning the Nobel Prize in Physics in 1903?",
#     "Marie Curie won the Nobel Prize in Physics in 1903 in Stockholm for her research on radioactivity by conducting experiments with uranium and radium.",
# ]

# docs = [
#     "Qui était l'astronaute qui a atterri sur la lune dans le cadre de la mission Apollo 11 ?",
#     "Quand Neil Armstrong a-t-il atterri sur la lune ?",
#     "Où Neil Armstrong a-t-il atterri dans le cadre de la mission Apollo 11 ?",
#     "Qu'est-ce que Neil Armstrong a accompli lors de la mission Apollo 11 ?",
#     "Pourquoi Neil Armstrong a-t-il atterri sur la lune ?",
#     "Comment Neil Armstrong a-t-il réussi à atterrir sur la lune ?",
#     "Le 20 juillet 1969, Neil Armstrong a atterri sur la lune pour accomplir la mission Apollo 11, qui a été réalisée grâce à la planification approfondie de la NASA et à la fusée Saturn V.",
# ]

# docs = [
#     "在阿波罗11号任务中，哪位宇航员登上了月球？",
#     "尼尔·阿姆斯特朗是什么时候登上月球的？",
#     "尼尔·阿姆斯特朗在阿波罗11号任务中在哪里着陆？",
#     "尼尔·阿姆斯特朗在阿波罗11号任务中取得了什么成就？",
#     "尼尔·阿姆斯特朗为什么登上月球？",
#     "尼尔·阿姆斯特朗是如何成功登上月球的？",
#     "1969年7月20日，尼尔·阿姆斯特朗登上月球，完成了阿波罗11号任务，这得益于NASA的精心规划和土星五号火箭。",
# ]


embeddings = model.encode(docs, normalize_embeddings=True)
query_embeddings = embeddings[:-1]
doc_embedding = embeddings[-1]
query_mean = query_embeddings.mean(axis=0)
query_mean = query_mean / np.linalg.norm(query_mean)
docs[-1] = 'Chunk'
docs = docs + ['Mean']
embeddings = np.vstack((query_embeddings,[doc_embedding, query_mean]))
similarities = model.similarity(embeddings, embeddings)

fig, axs = plt.subplots(1, 2, figsize=(16, 8))

axs[0].imshow(similarities, cmap='Blues')
for i in range(len(docs)):
    for j in range(len(docs)):
        axs[0].text(j, i, round(similarities[i, j].item(), 2), ha='center', va='center', color='black')
docs = [doc.split()[0] for doc in docs]
axs[0].set_xticks(range(len(docs)))
axs[0].set_xticklabels(docs)
axs[0].set_yticks(range(len(docs)))
axs[0].set_yticklabels(docs)
axs[0].set_title('Similarity matrix of sentence embeddings')

normal_vector = doc_embedding - query_mean
normal_vector = normal_vector / np.linalg.norm(normal_vector)
angles = []
for query_embedding in query_embeddings:
    plane_vector = query_embedding - query_mean
    plane_vector = plane_vector / np.linalg.norm(plane_vector)
    angle = np.arccos(np.dot(normal_vector, plane_vector))
    angle = np.degrees(angle)
    angles.append(angle)
print("Angles: ", angles)

import seaborn as sns
from scipy.stats import gaussian_kde
sns.set(style="whitegrid")
custom_colors = [
    (243/255, 135/255, 145/255), 
    (164/255, 234/255, 199/255),
    (161/255, 186/255, 216/255), 
    (160/255, 101/255, 147/255),  
    (164/255, 209/255, 208/255), 
]
density = gaussian_kde(angles)
xs = np.linspace(60, 110, 800)
ys = density(xs)
sns.lineplot(x=xs, y=ys, linewidth=2, color=custom_colors[0], label=model_name, ax=axs[1])
axs[1].fill_between(xs, ys, alpha=0.3, color=custom_colors[0])
axs[1].set_xlabel('Angle (Degrees)')
axs[1].set_ylabel('Density')
axs[1].grid(False)
axs[1].legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=True, shadow=True, fontsize=22, prop={'size': 22})
plt.tight_layout()
plt.savefig('demo_output.png', dpi=300)

# plt.figure(figsize=(16, 9))
# plt.rcParams.update({'font.size': 13})
# plt.imshow(similarities, cmap='Blues')
# for i in range(len(docs)):
#     for j in range(len(docs)):
#         plt.text(j, i, round(similarities[i, j].item(), 2), ha='center', va='center', color='black')
# docs = [doc.split()[0] for doc in docs]
# plt.xticks(range(len(docs)), docs)
# plt.yticks(range(len(docs)), docs)
# plt.title('Similarity matrix of sentence embeddings')
# # plt.show()
# plt.savefig('similarity_matrix.png')


# normal_vector = doc_embedding - query_mean
# normal_vector = normal_vector / np.linalg.norm(normal_vector)
# angles = []
# for query_embedding in query_embeddings:
#     plane_vector = query_embedding - query_mean
#     plane_vector = plane_vector / np.linalg.norm(plane_vector)
#     angle = np.arccos(np.dot(normal_vector, plane_vector))
#     angle = np.degrees(angle)
#     angles.append(angle)
# print(angles)

# import seaborn as sns
# from scipy.stats import gaussian_kde
# plt.clf()
# plt.figure(figsize=(16, 9))
# sns.set(style="whitegrid")
# plt.rcParams.update({
#     'font.size': 24,             # Global font size for all text
#     'axes.titlesize': 24,        # Font size for axes titles
#     'axes.labelsize': 24,        # Font size for x and y labels
#     'xtick.labelsize': 24,       # Font size for x tick labels
#     'ytick.labelsize': 24,       # Font size for y tick labels
#     'legend.fontsize': 22,       # Font size for legend
#     'legend.title_fontsize': 22  # Font size for legend title
# })
# custom_colors = [
#     (243/255, 135/255, 145/255), 
#     (164/255, 234/255, 199/255),
#     (161/255, 186/255, 216/255), 
#     (160/255, 101/255, 147/255),  
#     (164/255, 209/255, 208/255), 
# ]
# density = gaussian_kde(angles)
# xs = np.linspace(60, 110, 800)
# ys = density(xs)
# sns.lineplot(x=xs, y=ys, linewidth=2, color=custom_colors[0], label=model_name)
# plt.fill_between(xs, ys, alpha=0.3, color=custom_colors[0])
# plt.xlabel('Angle (Degrees)')
# plt.ylabel('Density')
# plt.grid(False)
# plt.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=True, shadow=True, fontsize=22, prop={'size': 22})
# plt.tight_layout()
# # plt.show()
# plt.savefig('angle_distribution.png', dpi=300)