def tool():
    title = "文本翻译"
    description = "将输入本文翻译成所选择的目标语言。"
    # 如果没有参数则留写空字典{}即可 
    # 参数请按照prompt中参数出现次序填写
    parameters = {
        "language": "翻译的目标语言",
        "text": "所需要翻译的文本"
    }
    prompt = f"""
    As an experimental translator,
    please translate the text delimited by triple backticks to {{language}}.
    
    The translation you make needs to conform to the convention of target language, such as late adverbial, object preposition, passive voice, etc.

    ```{{text}}```
    """
    return title, description, parameters, prompt
