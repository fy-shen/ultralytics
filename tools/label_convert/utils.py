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
        self.x = round((xtl + xbr) / 2 / width, 6)
        self.y = round((ytl + ybr) / 2 / height, 6)
        self.w = round(abs(xbr - xtl) / width, 6)
        self.h = round(abs(ybr - ytl) / height, 6)
