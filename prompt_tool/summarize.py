def tool():
    title = "文本总结"
    description = "将输入长本文进行总结。"
    # 如果没有参数则留写空字典{}即可
    # 参数请按照prompt中参数出现次序填写
    parameters = {
        "text": "需要进行总结的文本"
    }
    prompt = f"""
    As a good article editor,
    please summarize the text delimited by triple backticks naturally and concisely.

    You need summarize the text in the form of a declarative sentence.

    ```{{text}}```
    """
    return title, description, parameters, prompt
