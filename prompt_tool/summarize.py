def tool():
    title = "文本总结"
    description = "将输入长本文进行总结。"
    # 如果没有参数则留写空字典{}即可
    # 参数请按照prompt中参数出现次序填写
    parameters = {
        "text": "需要进行总结的文本"
    }
    prompt = f"""
    As a good text summarizer,
    please summarize the following text delimited by ``` naturally,\
    and the language of the summary needs to be the same as the language of the text.

    You should summarize the text in the form of a declarative sentence.

    ```{{text}}```
    """
    return title, description, parameters, prompt
