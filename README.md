# 大模型(LLMs)面试经验

更多面试内部资料，大模型实战经验请扫码加入下面的知识星球：

 <img src="http://image.rarelimiting.com/ef790b3b816bc0568a217ffa7e7e59f252b672b41c203a23dd304ee452d9d4ea.png" width = "450" height = "498" alt="图片名称" align=center />

【如果你曾经是我的学员，请添加微信aimaksen，告知你是哪里学员，退回星球费用】

## 大模型基础面试
### 1、目前主流的开源模型体系有哪些？
目前主流的开源模型体系 分三种：

第一种：prefix Decoder 系
介绍：输入双向注意力，输出单向注意力
代表模型：ChatGLM、ChatGLM2、U-PaLM
第二种：causal Decoder 系
介绍：从左到右的单向注意力
代表模型：LLaMA-7B、LLaMa 衍生物
第三种：Encoder-Decoder
介绍：输入双向注意力，输出单向注意力
代表模型：T5、Flan-T5、BART

![图 14](http://image.rarelimiting.com/4baa14c515834ae00a68682dc53d0023bd220ed27059b6f87776bb16cb0abd6c.png)  


### 2、prefix Decoder 和 causal Decoder 和 Encoder-Decoder 区别是什么？
     prefix Decoder 和 causal Decoder 和 Encoder-Decoder 区别 在于 attention mask不同：

Encoder-Decoder：
在输入上采用双向注意力，对问题的编码理解更充分
适用任务：在偏理解的 NLP 任务上效果好
缺点：在长文本生成任务上效果差，训练效率低；
causal Decoder：
自回归语言模型，预训练和下游应用是完全一致的，严格遵守只有后面的token才能看到前面的token的规则；
适用任务：文本生成任务效果好
优点：训练效率高，zero-shot 能力更强，具有涌现能力
prefix Decoder：
特点：prefix部分的token互相能看到，causal Decoder 和 Encoder-Decoder 折中；
缺点：训练效率低

![图 15](http://image.rarelimiting.com/f1d7122c25e3c5bc03d38f344967b71392cfdd90a49e86d584feb19c56e1f6a3.png)  


### 3、大模型LLM的 训练目标 是什么？
     
3.1. 语言模型

根据 已有词 预测下一个词，训练目标为最大似然函数：

![图 16](http://image.rarelimiting.com/9b84f673b93580a7781a3294e0d50fb93321eecefa55897e0489ba3629c67d1f.png)  


训练效率：Prefix Decoder < Causal Decoder

Causal Decoder 结构会在 所有 token 上计算损失，而 Prefix Decoder 只会在 输出上 计算损失。

3.2. 去噪自编码器

随机替换掉一些文本段，训练语言模型去恢复被打乱的文本段。目标函数为:

![图 17](http://image.rarelimiting.com/cb71ab13c41f5c16e7ac12594c77dd8ef4b1aa9609068c71610e6a75eba47705.png)  


去噪自编码器的实现难度更高。采用去噪自编码器作为训练目标的任务有GLM-130B、T5.

### 4、涌现能力是啥原因？
根据前人分析和论文总结，大致是2个猜想：

任务的评价指标不够平滑；
复杂任务 vs 子任务，这个其实好理解，比如我们假设某个任务 T 有 5 个子任务 Sub-T 构成，每个 sub-T 随着模型增长，指标从 40% 提升到 60%，但是最终任务的指标只从 1.1% 提升到了 7%，也就是说宏观上看到了涌现现象，但是子任务效果其实是平滑增长的。


### 5、为何现在的大模型大部分是Decoder only结构？
因为decoder-only结构模型在没有任何微调数据的情况下，zero-shot的表现能力最好。而encoder-decoder则需要在一定量的标注数据上做multitask-finetuning才能够激发最佳性能。

目前的Large LM的训练范式还是在大规模语料shang 做自监督学习，很显然zero-shot性能更好的decoder-only架构才能更好的利用这些无标注的数据。

大模型使用decoder-only架构除了训练效率和工程实现上的优势外，在理论上因为Encoder的双向注意力会存在低秩的问题，这可能会削弱模型的表达能力。就生成任务而言，引入双向注意力并无实质的好处。而Encoder-decoder模型架构之所以能够在某些场景下表现更好，大概是因为它多了一倍参数。所以在同等参数量、同等推理成本下，Decoder-only架构就是最优的选择了。

### 6、简单 介绍一下 大模型【LLMs】？
大模型：一般指1亿以上参数的模型，但是这个标准一直在升级，目前万亿参数以上的模型也有了。大语言模型（Large Language Model，LLM）是针对语言的大模型。

### 7、大模型【LLMs】后面跟的 175B、60B、540B等 指什么？
175B、60B、540B等：这些一般指参数的个数，B是Billion/十亿的意思，175B是1750亿参数，这是ChatGPT大约的参数规模。

### 8、大模型【LLMs】具有什么优点？
可以利用大量的无标注数据来训练一个通用的模型，然后再用少量的有标注数据来微调模型，以适应特定的任务。这种预训练和微调的方法可以减少数据标注的成本和时间，提高模型的泛化能力；
可以利用生成式人工智能技术来产生新颖和有价值的内容，例如图像、文本、音乐等。这种生成能力可以帮助用户在创意、娱乐、教育等领域获得更好的体验和效果；
可以利用涌现能力（Emergent Capabilities）来完成一些之前无法完成或者很难完成的任务，例如数学应用题、常识推理、符号操作等。这种涌现能力可以反映模型的智能水平和推理能力。

### 9、大模型【LLMs】具有什么缺点？
需要消耗大量的计算资源和存储资源来训练和运行，这会增加经济和环境的负担。据估计，训练一个GPT-3模型需要消耗约30万美元，并产生约284吨二氧化碳排放；
需要面对数据质量和安全性的问题，例如数据偏见、数据泄露、数据滥用等。这些问题可能会导致模型产生不准确或不道德的输出，并影响用户或社会的利益；
需要考虑可解释性、可靠性、可持续性等方面的挑战，例如如何理解和控制模型的行为、如何保证模型的正确性和稳定性、如何平衡模型的效益和风险等。这些挑战需要多方面的研究和合作，以确保大模型能够健康地发展。


## 大模型（LLMs）langchain 面

### 1、 什么是 LangChain?
LangChain是一个强大的框架，旨在帮助开发人员使用语言模型构建端到端的应用程序。它提供了一套工具、组件和接口，可简化创建由大型语言模型 (LLM) 和聊天模型提供支持的应用程序的过程。LangChain 可以轻松管理与语言模型的交互，将多个组件链接在一起，并集成额外的资源，例如 API 和数据库。



###2、 LangChain 包含哪些 核心概念？
2.1 LangChain 中 Components and Chains 是什么？
Component ：模块化的构建块，可以组合起来创建强大的应用程序；
Chain ：组合在一起以完成特定任务的一系列 Components（或其他 Chain）；
注：一个 Chain 可能包括一个 Prompt 模板、一个语言模型和一个输出解析器，它们一起工作以处理用户输入、生成响应并处理输出。


2.2 LangChain 中 Prompt Templates and Values 是什么？
Prompt Template 作用：负责创建 PromptValue，这是最终传递给语言模型的内容
Prompt Template 特点：有助于将用户输入和其他动态信息转换为适合语言模型的格式。PromptValues 是具有方法的类，这些方法可以转换为每个模型类型期望的确切输入类型（如文本或聊天消息）。


2.3 LangChain 中 Example Selectors 是什么？
作用：当您想要在 Prompts 中动态包含示例时，Example Selectors 很有用。他们接受用户输入并返回一个示例列表以在提示中使用，使其更强大和特定于上下文。


2.4 LangChain 中 Output Parsers 是什么？
作用： 负责将语言模型响应构建为更有用的格式
实现方法：
一种用于提供格式化指令
另一种用于将语言模型的响应解析为结构化格式
特点：使得在您的应用程序中处理输出数据变得更加容易。


2.5 LangChain 中 Indexes and Retrievers 是什么？
Index ：一种组织文档的方式，使语言模型更容易与它们交互；

Retrievers：用于获取相关文档并将它们与语言模型组合的接口；

注：LangChain 提供了用于处理不同类型的索引和检索器的工具和功能，例如矢量数据库和文本拆分器。



2.6 LangChain 中 Chat Message History 是什么？
Chat Message History 作用：负责记住所有以前的聊天交互数据，然后可以将这些交互数据传递回模型、汇总或以其他方式组合；
优点：有助于维护上下文并提高模型对对话的理解


2.7 LangChain 中 Agents and Toolkits 是什么？
Agent ：在 LangChain 中推动决策制定的实体。他们可以访问一套工具，并可以根据用户输入决定调用哪个工具；
Tookits ：一组工具，当它们一起使用时，可以完成特定的任务。代理执行器负责使用适当的工具运行代理。
通过理解和利用这些核心概念，您可以利用 LangChain 的强大功能来构建适应性强、高效且能够处理复杂用例的高级语言模型应用程序。



### 3、什么是 LangChain Agent?
介绍：LangChain Agent 是框架中驱动决策制定的实体。它可以访问一组工具，并可以根据用户的输入决定调用哪个工具；
优点：LangChain Agent 帮助构建复杂的应用程序，这些应用程序需要自适应和特定于上下文的响应。当存在取决于用户输入和其他因素的未知交互链时，它们特别有用。


### 4、如何使用 LangChain ?
要使用 LangChain，开发人员首先要导入必要的组件和工具，例如 LLMs, chat models, agents, chains, 内存功能。这些组件组合起来创建一个可以理解、处理和响应用户输入的应用程序。



### 5、LangChain 支持哪些功能?
针对特定文档的问答：根据给定的文档回答问题，使用这些文档中的信息来创建答案。
聊天机器人：构建可以利用 LLM 的功能生成文本的聊天机器人。
Agents：开发可以决定行动、采取这些行动、观察结果并继续执行直到完成的代理。


### 6、什么是 LangChain model?
LangChain model 是一种抽象，表示框架中使用的不同类型的模型。LangChain 中的模型主要分为三类：

LLM（大型语言模型）：这些模型将文本字符串作为输入并返回文本字符串作为输出。它们是许多语言模型应用程序的支柱。
聊天模型( Chat Model)：聊天模型由语言模型支持，但具有更结构化的 API。他们将聊天消息列表作为输入并返回聊天消息。这使得管理对话历史记录和维护上下文变得容易。
文本嵌入模型(Text Embedding Models)：这些模型将文本作为输入并返回表示文本嵌入的浮点列表。这些嵌入可用于文档检索、聚类和相似性比较等任务。
开发人员可以为他们的用例选择合适的 LangChain 模型，并利用提供的组件来构建他们的应用程序。



### 7、LangChain 包含哪些特点?
LangChain 旨在为六个主要领域的开发人员提供支持：

LLM 和提示：LangChain 使管理提示、优化它们以及为所有 LLM 创建通用界面变得容易。此外，它还包括一些用于处理 LLM 的便捷实用程序。
链(Chain)：这些是对 LLM 或其他实用程序的调用序列。LangChain 为链提供标准接口，与各种工具集成，为流行应用提供端到端的链。
数据增强生成：LangChain 使链能够与外部数据源交互以收集生成步骤的数据。例如，它可以帮助总结长文本或使用特定数据源回答问题。
Agents：Agents 让 LLM 做出有关行动的决定，采取这些行动，检查结果，并继续前进直到工作完成。LangChain 提供了代理的标准接口，多种代理可供选择，以及端到端的代理示例。
内存：LangChain 有一个标准的内存接口，有助于维护链或代理调用之间的状态。它还提供了一系列内存实现和使用内存的链或代理的示例。
评估：很难用传统指标评估生成模型。这就是为什么 LangChain 提供提示和链来帮助开发者自己使用 LLM 评估他们的模型。


### 8、LangChain 如何使用?
8.1 LangChain 如何调用 LLMs 生成回复？
Models: 指各类训练好大语言模型（eg: chatgpt(未开源)，chatglm，vicuna等）

```python
# 官方llm使用OPENAI 接口
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")
prompt = "你好"
response = llm(prompt)
# 你好，我是chatGPT,很高兴能够和你聊天。有什么我可以帮助你的吗？

# 我们用chatglm来演示该过程，封装一下即可
from transformers import AutoTokenizer, AutoModel
class chatGLM():
    def __init__(self, model_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda().eval()

    def __call__(self, prompt) -> Any:
        response, _ = self.model.chat(self.tokenizer , prompt) # 这里演示未使用流式接口. stream_chat()
        return response

llm =  chatGLM(model_name="THUDM/chatglm-6b")
prompt = "你好"
response = llm(prompt)
print("response: %s"%response)
“”“
response: 你好 ！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
”“”
```


8.2 LangChain 如何修改 提示模板？
langchain.PromptTemplate: langchain中的提示模板类

根据不同的下游任务设计不同的prompt模板，然后填入内容，生成新的prompt。目的其实就是为了通过设计更准确的提示词，来引导大模型输出更合理的内容。

```python
from langchain import PromptTemplate
template = """
Explain the concept of {concept} in couple of lines
"""
prompt = PromptTemplate(input_variables=["concept"], template=template)
prompt = prompt.format(concept="regularization")
print(“prompt=%s”%prompt)
#'\nExplain the concept of regularization in couple of lines\n'

-------------------------
template = "请给我解释一下{concept}的意思"
prompt = PromptTemplate(input_variables=["concept"], template=template)
prompt = prompt.format(concept="人工智能")
print(“prompt=%s”%prompt)
#'\n请给我解释一下人工智能的意思\n'
```

8.3 LangChain 如何链接多个组件处理一个特定的下游任务？
```python
#chains ---------
from langchain.chains import LLMChain
chain = LLMChain(llm=openAI(), prompt=promptTem)
print(chain.run("你好"))

#chains ---------Chatglm对象不符合LLMChain类llm对象要求，模仿一下
class DemoChain():
    def __init__(self, llm, prompt) -> None:
        self.llm = llm
        self.prompt = prompt

    def run(self, query) -> Any:
        prompt = self.prompt.format(concept=query)
        print("query=%s  ->prompt=%s"%(query, prompt))
        response = self.llm(prompt) 
        return response
    
chain = DemoChain(llm=llm, prompt=promptTem)
print(chain.run(query="天道酬勤"))

“”“
query=天道酬勤  ->prompt=请给我解释一下天道酬勤的意思
天道酬勤是指自然界的规律认为只要一个人勤奋努力，就有可能会获得成功。这个成语的意思是说，尽管一个人可能需要付出很多努力才能取得成功，但只要他/她坚持不懈地努力，就有可能会得到回报。
”“”
```

8.4 LangChain 如何Embedding & vector store？
Emebdding这个过程想必大家很熟悉，简单理解就是把现实中的信息通过各类算法编码成一个高维向量，便于计算机快速计算。

DL的语言模型建模一般开头都是word embedding，看情况会加position embedding。比如咱们的LLM的建模
常规检索一般是把refernce数据都先Embedding入库，服务阶段query进来Embedding后再快速在库中查询相似topk。比如langchain-chatGLM的本地知识库QA系统的入库和检测过程。
多模态的方案：同时把语音，文字，图片用不同的模型做Embedding后，再做多模态的模型建模和多模态交互。比如这两天的Visual-chatGLM.
#官方示例代码，用的OpenAI的ada的文本Embedding模型
#1） Embeding model
```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model_name="ada")
query_result = embeddings.embed_query("你好")
```

#2) 文本切割
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=0
)
texts = """天道酬勤”并不是鼓励人们不劳而获，而是提醒人们要遵循自然规律，通过不断的努力和付出来追求自己的目标。\n这种努力不仅仅是指身体上的劳动，
也包括精神上的努力和思考，以及学习和适应变化的能力。\n只要一个人具备这些能力，他就有可能会获得成功。"""
texts = text_splitter.create_documents([texts])
print(texts[0].page_content)
```

# 3)入库检索，官方使用的Pinecone,他提供一个后台管理界面 | 用户需求太大，不好用了已经，一直加载中....
```python
import pinecone
from langchain.vectorstores import Pinecone
pinecone.init(api_key=os.getenv(""), enviroment=os.getenv(""))

index_name = "demo"
search = Pinecone.from_documents(texts=texts, embeddings, index_name=index_name)
query = "What is magical about an autoencoder?"
result = search.similarity_search(query)

#---------------------------------------------这里参考langchain-chatglm代码
# 1） Embedding model:  text2vec-large-chinese
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese",
                                    model_kwargs={'device': "cuda"})
query_result = embeddings.embed_query("你好")

#2)文本分割， 这里仅为了方便快速看流程，实际应用的会复杂一些
texts = """天道酬勤”并不是鼓励人们不劳而获，而是提醒人们要遵循自然规律，通过不断的努力和付出来追求自己的目标。\n这种努力不仅仅是指身体上的劳动，
也包括精神上的努力和思考，以及学习和适应变化的能力。\n只要一个人具备这些能力，他就有可能会获得成功。"""
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
class TextSpliter(CharacterTextSplitter):
    def __init__(self, separator: str = "\n\n", **kwargs: Any):
        super().__init__(separator, **kwargs)
    def split_text(self, text: str) -> List[str]:

        texts = text.split("\n")
        texts = [Document(page_content=text, metadata={”from“: "知识库.txt"}) for text in texts]
        return texts
    
text_splitter = TextSpliter()
texts = text_splitter.split_text(texts)
```

#3) 直接本地存储
```python
vs_path = "./demo-vs"
from langchain.vectorstores import FAISS
docs = embeddings.embed_documents(sentences)
vector_store = FAISS.from_documents(texts, embeddings)
vector_store.save_local(vs_path)

vector_store = FAISS.load_local(vs_path, embeddings)
related_docs_with_score = vector_store.similarity_search_with_score(query, k=2)
```

### 9、LangChain 存在哪些问题及方法方案？
1. LangChain 低效的令牌使用问题
问题：Langchain的一个重要问题是它的令牌计数功能，对于小数据集来说，它的效率很低。虽然一些开发人员选择创建自己的令牌计数函数，但也有其他解决方案可以解决这个问题。
解决方案：Tiktoken是OpenAI开发的Python库，用于更有效地解决令牌计数问题。它提供了一种简单的方法来计算文本字符串中的令牌，而不需要使用像Langchain这样的框架来完成这项特定任务。
2. LangChain 文档的问题
问题：文档是任何框架可用性的基石，而Langchain因其不充分且经常不准确的文档而受到指责。误导性的文档可能导致开发项目中代价高昂的错误，并且还经常有404错误页面。这可能与Langchain还在快速发展有关，作为快速的版本迭代，文档的延后性问题
3. LangChain 太多概念容易混淆，过多的“辅助”函数问题
问题：Langchain的代码库因很多概念让人混淆而备受批评，这使得开发人员很难理解和使用它。这种问题的一个方面是存在大量的“helper”函数，仔细检查就会发现它们本质上是标准Python函数的包装器。开发人员可能更喜欢提供更清晰和直接访问核心功能的框架，而不需要复杂的中间功能。
简单的分割函数：


### 10、LangChain 行为不一致并且隐藏细节问题
问题：LangChain因隐藏重要细节和行为不一致而受到批评，这可能导致生产系统出现意想不到的问题。
eg: Langchain ConversationRetrievalChain的一个有趣的方面，它涉及到输入问题的重新措辞。这种重复措辞有时会非常广泛，甚至破坏了对话的自然流畅性，使对话脱离了上下文。

LangChain 缺乏标准的可互操作数据类型问题
问题：缺乏表示数据的标准方法。这种一致性的缺乏可能会阻碍与其他框架和工具的集成，使其在更广泛的机器学习工具生态系统中工作具有挑战性。
LangChain 替代方案？
是否有更好的替代方案可以提供更容易使用、可伸缩性、活动性和特性。

LlamaIndex是一个数据框架，它可以很容易地将大型语言模型连接到自定义数据源。它可用于存储、查询和索引数据，还提供了各种数据可视化和分析工具。
Deepset Haystack是另外一个开源框架，用于使用大型语言模型构建搜索和问答应用程序。它基于Hugging Face Transformers，提供了多种查询和理解文本数据的工具。


## [大模型（LLMs）进阶面](https://articles.zsxq.com/id_4wsf9thdioq3.html)

1. LLMs 复读机问题
   1. 什么是 LLMs 复读机问题？
   2. 为什么会出现 LLMs 复读机问题？
   3. 如何缓解 LLMs 复读机问题？
2. llama 系列问题
   1. llama 输入句子长度理论上可以无限长吗？
3. 什么情况用Bert模型，什么情况用LLaMA、ChatGLM类大模型，咋选？
4. 各个专业领域是否需要各自的大模型来服务？
5. 如何让大模型处理更长的文本？
6. ...

- [点击查看答案](https://articles.zsxq.com/id_4wsf9thdioq3.html)


## [大模型（LLMs）微调面](https://articles.zsxq.com/id_x0u0jde3jf7k.html)

1. 如果想要在某个模型基础上做全参数微调，究竟需要多少显存？
2. 为什么SFT之后感觉LLM傻了?
3. SFT 指令微调数据 如何构建?
4. 领域模型Continue PreTrain 数据选取？
5. 领域数据训练后，通用能力往往会有所下降，如何缓解模型遗忘通用能力？
6. 领域模型Continue PreTrain ，如何 让模型在预训练过程中就学习到更多的知识？
7. 进行SFT操作的时候，基座模型选用Chat还是Base?
8. 领域模型微调 指令\&数据输入格式 要求？
9. 领域模型微调 领域评测集 构建？
10. 领域模型词表扩增是不是有必要的？
11. 如何训练自己的大模型？
12. 训练中文大模型有啥经验？
13. 指令微调的好处？
14. 预训练和微调哪个阶段注入知识的？
15. 想让模型学习某个领域或行业的知识，是应该预训练还是应该微调？
16. 多轮对话任务如何微调模型？
17. 微调后的模型出现能力劣化，灾难性遗忘是怎么回事？
18. 微调模型需要多大显存？
19. 大模型LLM进行SFT操作的时候在学习什么？
20. 预训练和SFT操作有什么不同
21. 样本量规模增大，训练出现OOM错
22. 大模型LLM进行SFT 如何对样本进行优化？
23. 模型参数迭代实验
24. 微调大模型的一些建议
25. ...

- [点击查看答案](https://articles.zsxq.com/id_x0u0jde3jf7k.html)
