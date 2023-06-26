import re

text = "Sample text R123C456 Another"

pattern = r"R[0-9]{3}C[0-9]{3}"  # 匹配 R 后面跟着三位数字，然后是 C 后面跟着三位数字的字符串

matche = re.findall(pattern, text)  # 找到所有匹配的字符串

if matches:
    print("找到匹配的字符串:")
    for match in matches:
        print(match)
else:
    print("未找到匹配的字符串")