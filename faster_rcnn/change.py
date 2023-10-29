#good 02 bad 13

import xml.etree.ElementTree as ET
import os

mapping_dict = {'0':'0', '2': '0', '1':'1', '3':'1'}

for xml in os.listdir("./"):
    if xml.endswith("xml"):
        mytree = ET.parse(os.path.join('./',xml))
        myroot = mytree.getroot()
        for object_ in myroot.findall('.//object'):
            name = object_.find('name')

            print('开始处理{}文件'.format(xml))
            name.text = mapping_dict[str(name.text)]

        mytree.write(xml)
    print('已替换{}文件'.format(xml))

