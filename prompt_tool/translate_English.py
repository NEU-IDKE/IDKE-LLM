def tool():
    title = "英文翻译"
    description = "通过输入语言和文本，将英文翻译成其他语言，来完成翻译工作。"
    # 如果没有参数则留写空字典{}即可 
    # 参数请按照prompt中参数出现次序填写
    paramters = {
                "language":"输出的目标语言",
                "text":"翻译文本"
               }
    prompt = f"""
            Translate the following English text to {{language}}: \ 
            ```{{text}}```
            """
    return title, description, paramters, prompt