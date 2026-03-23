import os
import xml.etree.ElementTree as ET


class CVATLoader:
    def __init__(self, xml):
        self.tree = ET.parse(xml)
        self.root = self.tree.getroot()
        self._load_tasks()

    def _load_tasks(self):
        self.tasks = {}
        tasks = self.root.find('meta').find('project').find('tasks').findall('task')
        for task in tasks:
            self.tasks[task.findtext('id')] = {
                'name': task.findtext('name'),
                'source': task.findtext('source'),
            }


class CVATImg:
    def __init__(self, img):
        self.idx = int(img.attrib['name'].split('_')[-1])
        self.task_id = img.attrib['task_id']
        self.w = float(img.attrib['width'])
        self.h = float(img.attrib['height'])
        self.boxes = img.findall('box')


class CVATBox:
    def __init__(self, box, width, height):
        self.label = box.attrib['label'].lower()
        xtl, ytl, xbr, ybr = map(
            float, [box.attrib['xtl'], box.attrib['ytl'], box.attrib['xbr'], box.attrib['ybr']]
        )
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr

        self.x = (xtl + xbr) / 2
        self.y = (ytl + ybr) / 2
        self.w = abs(xbr - xtl)
        self.h = abs(ybr - ytl)

        self.xn = self.x / width
        self.yn = self.y / height
        self.wn = self.w / width
        self.hn = self.h / height
