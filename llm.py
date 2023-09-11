# 导入包
import streamlit as st
import openai
import numpy as np
from io import StringIO
import time
import os
import importlib

# 在requirement需要添加的额外的库
import tiktoken
from streamlit_option_menu import option_menu



def page1():
    st.title("🐜 IDKE-LLM")
    with st.expander('**Welcome to 🐜 IDKE-LLM!**'):
        note = """
       这个网页是用于存储 IDKE 实验室有关大模型的各种学习成果，仅供学习交流使用。
        
       本实验室采用当前的大模型与各种微调的大模型（以下简称“模型”）技术。
       但是在使用模型时，请您务必注意以下事项：
         1.信息准确：模型生成的内容可能会受到输入数据、上下文以及其他因素的影响。但不能保证所有输出都是完全无误的。
         2.内容审核：请在使用模型生成的内容前，自行进行审查和验证。尤其是涉及到重要信息、专业建议等方面，务必谨慎对待。
         3.合法合规：请确保您使用模型的方式符合所有适用的法律法规和规定。模型可能会生成各种类型的内容，但最终的责任仍然由用户承担。
         4.隐私保护：请不要在与模型的交互中透露任何个人敏感信息或隐私内容。尽管我们将不会存储或使用用户输入的信息。
         5.用户反馈：我们欢迎用户提供有关模型输出的反馈，以帮助我们持续改进和优化服务质量。
         6.技术限制：模型仍然受到技术限制，可能会产生一些不符合实际情况的输出。请理解并谅解这一点。
            
       使用本网页即表示您同意并接受以上免责声明。如果您对免责声明有任何疑问或需要进一步的解释，请随时联系我们。
       联系方式: drasick59596@163.com
        """
        st.text(note)
        
    with st.expander('**关于聊天对话**'):
        st.text('聊天对话功能是指利用 API 接口通过 ChatGPT 进行交互的能力。\n通过向模型发送消息文本，用户能够实现与 ChatGPT 的动态对话。\n请在左侧菜单输入您的 OpenAI API 进行使用。')
        
    with st.expander('**关于角色扮演**'):
        st.text('角色扮演特性允许用户选择不同的角色扮演指令，在特定的情境下与大模型进行对话。')
        st.text('请按照以下格式，保存成 python 文件，并上传到 GitHub 的 prompt_character 文件夹下。\n并将你的title添加到该文件夹中的characters.txt文件中。')
        code = """
        def character():
            title = "赛博侦探"
            description = "该人格扮演一个侦探，通过和你对话来推测谁是凶手。"
            context = [
                {
                    'role':'system',
                    'content':\"\"\"
                        你是一个侦探，能够通过事件线索、时间线、嫌疑人的证词来推理出谁是凶手。
                        1.首先询问人类如何称呼，并咨询是否需要案件的帮助；
                        2.……
                        3.……
                        7.结合每个嫌疑人的杀人动机和作案可能性，推断最后的凶手是谁，并给出理由。

                        <注意>……
                        <注意>……
                        \"\"\"
                }
            ] 
            return title, description, context
            """
        st.code(code)
        st.download_button("下载人格模板",data= code ,file_name = "修改为人格名称.py")
        
        
    with st.expander('**关于提示词工具包**'):
        st.text('提示词工具包提供了一系列功能性提示词汇。\n这些预设提示词汇具备一定的实用性，用户可以在与 ChatGPT 的交互中灵活应用。')
        st.text('请按照以下格式，保存成 python 文件，并上传到 GitHub 的 prompt_tool 文件夹下。\n并将你的title添加到该文件夹中的tools.txt文件中。')
        code = """
        def tool():
            title = "英文翻译"
            description = "通过输入语言和文本来完成翻译工作。"
            # 如果没有参数则留写空字典{}即可 
            # 参数请按照 prompt 中参数出现次序填写
            # 【重要】parameter 中的需要输入长文本的请将变量名称定义为 text！
            # 需要输入多个长文本的请将变量定义成 text_animal text_2 这种含 text 的格式
            paramters = {
                        "language":"输出的目标语言",
                        "text":"翻译文本"
                       }
            # f-string 格式的 prompt , 涉及的参数部分请用两个花括号 {{xxx}}
            prompt = f\"\"\"
                    Translate the following English text to {{language}}: \ 
                    ```{{text}}```
                    \"\"\"
            return title, description, paramters, prompt
               """
           
        st.code(code)
        st.download_button("下载提示词模板",data= code ,file_name = "修改为提示词名称.py")
        
    with st.expander('**关于模型微调**'):
        st.text('提供了多种微调大型模型的方法指导。\n这使得用户能够根据特定需求对模型的行为进行定制化调整。\n通过微调，用户可以调整模型的响应，使其更加符合特定应用场景或任务的要求。')
        
        st.text('请将学习记录保存成 Markdown 格式文件，并上传到 GitHub 的 fine_tuning 文件夹下。')
        st.text('【重要】\n如果 md 文件中有图片，请把图片格式所在行的 md 代码修改为【相对路径】的形式并一起上传！\n例如👉 ![image-XX](./image-XX.png)图片格式可以为 png 或者 jpg ')
        
    with st.expander('**关于强化学习**'):
        st.text('提供了多种大型模型强化学习的方法指导。\n强化学习指引用户如何通过奖励机制来调整大型模型的输出，从而使其更符合人类的期望。\n通过提供反馈和奖励，用户可以逐步训练模型，使其生成更为准确和符合预期的响应。\n这一特性可以有效地提升模型的性能，使其在实际应用中表现更为优异。')
        st.text('请将学习记录保存成 Markdown 格式文件，并上传到 GitHub 的 reinforcement learning 文件夹下。')
        st.text('【重要】\n如果 md 文件中有图片，请把图片格式所在行的 md 代码修改为【相对路径】的形式并一起上传！\n例如👉 ![image-XX](./image-XX.png) 图片格式可以为 png 或者 jpg ')
        
        
        
        
        
    
def page2():
    st.title("💬 Chatbot") 

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "你好，有什么需要帮忙的？"}]
    if "messages_length" not in st.session_state:
        st.session_state["messages_length"] = 0

    my_bar.progress(st.session_state.messages_length / 4050,text = f"⚡Token Usage(current page): {st.session_state.messages_length}") 
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("输入文本"):
        if not openai_api_key:
            st.info("❄️ 请输入你的 OpenAI API key")
            st.stop()

        openai.api_key = openai_api_key
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages_length += len(encoding.encode(prompt))
        if st.session_state.messages_length > 4050:
            st.info("❄️ 单轮对话结束")
            st.session_state["messages"] = [{"role": "assistant", "content": "你好，有什么需要帮忙的？"}]
            st.session_state["messages_length"] = 0
            my_bar.progress(0,text = f"⚡Token Usage(current page): 0")
            st.stop()
        my_bar.progress(st.session_state.messages_length/4050,text = f"⚡Token Usage(current page): {st.session_state.messages_length}") 
        st.chat_message("user").write(prompt)
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        msg = response.choices[0].message
        msg_length = int(response.usage.completion_tokens)
        st.session_state.messages.append(msg)
        st.session_state.messages_length += msg_length
        if st.session_state.messages_length > 4050:
            st.info("❄️ 单轮对话结束")
            st.session_state["messages"] = [{"role": "assistant", "content": "你好，有什么需要帮忙的？"}]
            st.session_state["messages_length"] = 0
            my_bar.progress(0,text = f"⚡Token Usage(current page): 0")
            st.stop()
        my_bar.progress(st.session_state.messages_length / 4050,text = f"⚡Token Usage(current page): {st.session_state.messages_length}") 
        st.chat_message("assistant").write(msg.content)
        
        
        
        
        
        
        
def page3():
    st.title("👤 Character")
    
    if "messages_c" not in st.session_state:
        st.session_state["messages_c"] = []
    if "messages_length_c" not in st.session_state:
        st.session_state["messages_length_c"] = 0
    
    if "character_index" not in st.session_state:
        st.session_state["character_index"] = 0
        
    my_bar.progress(st.session_state.messages_length_c / 4050,text = f"⚡Token Usage(current page): {st.session_state.messages_length_c}") 
               
    with st.expander('**设定对话角色**'):
        characters = {'<未选择>':"none"}
        # 打开文件
        with open('prompt_character/characters.txt', 'r',encoding='utf-8') as file:
            # 逐行读取内容
            for line in file:
                characters[line.split()[0]] = line.split()[1]
        if st.session_state.character_index != 0:        
            choose_character = st.selectbox('当前可用的人物设定：', [character for character in characters.keys()],index = st.session_state.character_index,disabled = True) 
            st.caption("🔒如需更改请结束当前对话")
        else:
            choose_character = st.selectbox('当前可用的人物设定：', [character for character in characters.keys()],index = st.session_state.character_index) 
        if choose_character != "<未选择>":
            module = importlib.import_module("prompt_character." + characters[choose_character][:-3])  # 去掉文件名后缀 '.py'
            title, description, context = module.character()  # 调用函数
            st.divider()
            st.subheader(title)
            st.write(description)
            
        st.session_state.character_index = [character for character in characters.keys()].index(choose_character)
            
    for i,msg in enumerate(st.session_state.messages_c):
        if i == 0:
            continue
        if msg["role"] == 'assistant':
            st.chat_message(msg["role"],avatar="👤").write(msg["content"])   
        else:
            st.chat_message(msg["role"],avatar="😶").write(msg["content"])   
#         st.write(choose_character)
    if prompt := st.chat_input("输入文本"):
        if not openai_api_key:
            st.info("❄️ 请输入你的 OpenAI API key")
            st.stop()
        if choose_character == '<未选择>':
            st.info("❄️ 请选择对话角色")
            st.stop()
            
        openai.api_key = openai_api_key
        
        # 添加角色设定
        if len(st.session_state.messages_c) == 0:
            st.session_state.messages_c.append(context[0])
            
        st.session_state.messages_c.append({"role": "user", "content": prompt})
        st.session_state.messages_length_c += len(encoding.encode(prompt))
        if st.session_state.messages_length_c > 4050:
            st.info("❄️ 单轮对话结束")
            st.session_state["messages_c"] = []
            st.session_state["messages_length_c"] = 0
            my_bar.progress(0,text = f"⚡Token Usage(current page): 0")
            st.stop()
        my_bar.progress(st.session_state.messages_length_c / 4050,text = f"⚡Token Usage(current page): {st.session_state.messages_length_c}") 
        st.chat_message("user",avatar="😶").write(prompt)
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages_c,temperature=0)
        msg = response.choices[0].message
        msg_length = int(response.usage.completion_tokens)
        st.session_state.messages_c.append(msg)
        st.session_state.messages_length_c += msg_length
        if st.session_state.messages_length_c > 4050:
            st.info("❄️ 单轮对话结束")
            st.session_state["messages_c"] = []
            st.session_state["messages_length_c"] = 0
            my_bar.progress(0,text = f"⚡Token Usage(current page): 0")
            st.stop()
        my_bar.progress(st.session_state.messages_length_c / 4050,text = f"⚡Token Usage(current page): {st.session_state.messages_length_c}") 
        st.chat_message("assistant",avatar="👤").write(msg.content)
        
        
        
        
        
def page4():
    st.title("📦 Prompt Tools")   
    
    st.session_state["messages_t"] = []
    st.session_state["messages_length_t"] = 0
        
    if "tool_index" not in st.session_state:
        st.session_state["tool_index"] = 0
        
    # 判断是否有可以提交的标记    
    if not "valid_inputs_received" in st.session_state:
        st.session_state["valid_inputs_received"] = False
    if not "paramters" in st.session_state:
        st.session_state["paramters"] = {}
        
    if not "paramters_values" in st.session_state:
        st.session_state["paramters_values"] = {}
        
    my_bar.progress(st.session_state.messages_length_t / 4050,text = f"⚡Token Usage(current page): {st.session_state.messages_length_t}")    
        
    with st.expander('**选择功能包**'):
        tools = {'<未选择>':"none"}
        # 打开文件
        with open('prompt_tool/tools.txt', 'r',encoding='utf-8') as file:
            # 逐行读取内容
            for line in file:
                tools[line.split()[0]] = line.split()[1]
            choose_tool = st.selectbox('当前可用的功能包：', [tool for tool in tools.keys()],index = st.session_state.tool_index) 
        if choose_tool != "<未选择>":
            module = importlib.import_module("prompt_tool." + tools[choose_tool][:-3])  # 去掉文件名后缀 '.py'
            title, description, paramters, prompt = module.tool()  # 调用函数
            st.divider()
            st.subheader(title)
            st.caption(description)
            st.write("参数需求")
            st.write(paramters)
            st.session_state.paramters = paramters
        else:
            st.session_state.paramters = {}
        st.session_state.tool_index = [tool for tool in tools.keys()].index(choose_tool)
        
    with st.form(key="my_form"):
        paramters_values = {}
        if st.session_state.paramters_values:
            for param in st.session_state.paramters:
                if "text" in param:
                    paramters_values[param] = st.text_area(param, st.session_state.paramters_values[param], help = st.session_state.paramters[param], height = 200)
                else:
                    paramters_values[param] = st.text_input(param, st.session_state.paramters_values[param], help = st.session_state.paramters[param])
        else:
            for param in st.session_state.paramters:
                if "text" in param:
                    paramters_values[param] = st.text_area(param, help = st.session_state.paramters[param], height = 200)
                else:
                    paramters_values[param] = st.text_input(param, help = st.session_state.paramters[param])
#         st.write(prompt)
        if not st.session_state.paramters:
            submit_button = st.form_submit_button(label="提交", disabled = True)
        else:
            submit_button = st.form_submit_button(label="提交" )
            
    st.session_state.paramters_values = paramters_values
    # 遍历判定所有参数是否初始化
    for param in paramters_values:
        if not paramters_values[param] and submit_button:
            st.toast(f"🚨 没有输入{param} !")
            st.session_state.valid_inputs_received = False
            st.stop()
    # 说明所有参数都已经完成赋值
    if len(paramters_values) != 0:
        st.session_state.valid_inputs_received = True  
    if submit_button and st.session_state.valid_inputs_received:
        if not openai_api_key:
            st.toast("❄️ 请输入你的 OpenAI API key")
            st.stop()
        openai.api_key = openai_api_key
#         st.write(prompt.format(**paramters_values))
        st.session_state.messages_t.append({"role": "user", "content": prompt.format(**paramters_values)})
        st.session_state.messages_length_t += len(encoding.encode(prompt))
        if st.session_state.messages_length_t > 4050:
            st.info("❄️ 输入文本过长")
            st.session_state["messages_t"] = []
            st.session_state["messages_length_t"] = 0
            my_bar.progress(0,text = f"⚡Token Usage(current page): 0")
            st.stop()
        my_bar.progress(st.session_state.messages_length_t / 4050,text = f"⚡Token Usage(current page): {st.session_state.messages_length_t}") 
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages_t,temperature = 0)
        msg = response.choices[0].message
        msg_length = int(response.usage.completion_tokens)
        st.session_state.messages_t.append(msg)
        st.session_state.messages_length_t += msg_length
        if st.session_state.messages_length_t > 4050:
             my_bar.progress(4050 / 4050,text = f"⚡Token Usage(current page): {st.session_state.messages_length_t}")
        else:
            my_bar.progress(st.session_state.messages_length_t / 4050,text = f"⚡Token Usage(current page): {st.session_state.messages_length_t}")
        with st.expander(':green[**查看结果**]',expanded = True):  
            st.write(msg.content)
            st.divider()
            st.download_button('下载输出的文本', msg.content, file_name='result.txt')
        st.toast(f"🎉 完成啦！")
        st.session_state["messages_t"] = []
        st.session_state["messages_length_t"] = 0
    # test
    st.session_state["paramters"] = {}
    st.session_state["paramters_values"] = {}
    st.session_state["valid_inputs_received"] = False
 
        
        
        
        
        
def page5():
    st.title("🔧 Fine Tuning", help = "🔒当前仅有学习资料，其他功能敬请期待") 
    if "fine_index" not in st.session_state:
        st.session_state["fine_index"] = 0
        
    with st.expander('**选择学习资料**'):
        materials = ['<未选择>']
        folder_path = 'fine_tuning'
        folders = os.listdir(folder_path)
        materials.extend(folders)
        choose_folder = st.selectbox('当前整理的学习资料：', materials, index = st.session_state.fine_index)
        
    if choose_folder != '<未选择>':
        inner_folder = os.path.join(folder_path, choose_folder)
        items = os.listdir(inner_folder)
        for item in items:
            if ".md" in item:
                item_path = os.path.join(inner_folder, item)
                with open(item_path, 'r', encoding = 'utf-8') as file:
                    split_article = ""
                    line = file.readline()
                    while line:
                        
                        if ".png" in line or ".jpg" in line:
                            st.write(split_article)
                            image_path = os.path.join(inner_folder, line.split("./")[-1][:-2])
                            image_path = image_path.replace("\\","/")
                            st.image(image_path)
                            split_article = ""
                        else:
                            split_article += line + '\n'
                        line = file.readline()
                    if split_article != "":
                        st.write(split_article)
                        
    st.session_state.fine_index = materials.index(choose_folder)
    
    
    
    
#     for folder in folders:
#         inner_folder = os.path.join(folder_path, folder)
#         items = os.listdir(inner_folder)
#         for item in items:
#             st.write(item)
            
    
    
def page6():
    st.title("🎰 Reinforcement Learning", help = "🔒当前仅有学习资料，其他功能敬请期待") 
    
    if "rein_index" not in st.session_state:
        st.session_state["rein_index"] = 0
        
    with st.expander('**选择学习资料**'):
        materials = ['<未选择>']
        folder_path = 'reinforcement_learning'
        folders = os.listdir(folder_path)
        materials.extend(folders)
        choose_folder = st.selectbox('当前整理的学习资料：', materials, index = st.session_state.rein_index)
        
    if choose_folder != '<未选择>':
        inner_folder = os.path.join(folder_path, choose_folder)
        items = os.listdir(inner_folder)
        for item in items:
            if ".md" in item:
                item_path = os.path.join(inner_folder, item)
                with open(item_path, 'r', encoding = 'utf-8') as file:
                    split_article = ""
                    line = file.readline()
                    while line:
                        
                        if ".png" in line or ".jpg" in line:
                            st.write(split_article)
                            image_path = os.path.join(inner_folder, line.split("./")[-1][:-2])
                            image_path = image_path.replace("\\","/")
                            st.image(image_path)
                            split_article = ""
                        else:
                            split_article += line + '\n'
                        line = file.readline()
                    if split_article != "":
                        st.write(split_article)
                        
    st.session_state.rein_index = materials.index(choose_folder)
    
    
        
# 设置网页标题
st.set_page_config(page_title="IDKE-LLM", page_icon="🐜")

# 设定文件路径
# st.session_state["url"] = f"https://github.com/NEU-IDKE/IDKE-LLM/blob/main/{file_name}?raw=true"
    
# 设置侧栏
with st.sidebar:
    selected = option_menu(
        "  🐜 IDKE - LLM",
        ["使用介绍", "聊天对话", "角色扮演","提示词工具包", "模型微调", "强化学习"],
        icons=["bi bi-book", "bi bi-chat-left-dots", "bi bi-robot","bi bi-brightness-alt-high","bi bi-bounding-box","bi bi-cpu"],
        menu_icon="bi bi-arrow-right",
        default_index=0,
    )
    openai_api_key = st.text_input("🔑 OpenAI API", key="chatbot_api_key", type="password", help="https://platform.openai.com/account/api-keys")
    my_bar = st.progress(0,text = "⚡Token Usage(current page): 0")
    if st.button(':red[结束当前]', help="结束对话，或清除右侧所有输入框"):
        if selected == "聊天对话":
            st.toast("单轮对话结束",icon='💥')
            st.session_state["messages"] = [{"role": "assistant", "content": "你好，有什么需要帮忙的？"}]
            st.session_state["messages_length"] = 0
        elif selected == "角色扮演":
            st.toast("单轮对话结束",icon='💥')
            st.session_state["messages_c"] = []
            st.session_state["messages_length_c"] = 0
            st.session_state["character_index"] = 0  
        elif selected == "提示词工具包":
            st.toast("清除所有输入框",icon='💥')
            st.session_state["messages_t"] = []
            st.session_state["messages_length_t"] = 0
            st.session_state["paramters_values"] = {}
            
       
    choose_model = st.selectbox('💡Choose Model', ['gpt-3.5-turbo','text-davinci-003', 'llama2-chat','Chat-GLM'], help = "🔒敬请期待")
    
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

if selected == "使用介绍":
    page1()
elif selected == "聊天对话":
    page2()
elif selected == "角色扮演":
    page3()    
elif selected == "提示词工具包":
    page4() 
elif selected == "模型微调":
    page5() 
elif selected == "强化学习":
    page6() 

