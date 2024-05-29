import cv2
import matplotlib.pyplot as plt
from PIL import Image

import time
import torch
import numpy as np
import matplotlib.patches as patches

from darknet import Darknet

def load_class_names(namesfile):
    
    # Создаем список для имен классов
    class_names = []
    
    # Открываем файл, который содержит имена классов
    with open(namesfile, 'r') as fp:
        # Файл содержит имена классов построчно
        lines = fp.readlines()
    
    # Проходимся по каждой линии из списки линий 
    for line in lines:
        # Сохраняем копию линии без пустот
        line = line.rstrip()
        
        # Записываем название класса в список созданный раннее
        class_names.append(line)
        
    return class_names

# Расположение cfg файла
cfg_file = 'data/cfg/crop_weed.cfg'

# Расположение весов модели
weight_file = 'data/weights/' + 'crop_weed_detection.weights'

# Расположение файла с имена классов
namesfile = 'data/names/obj.names'

# Загрузим архитектуру сети
m = Darknet(cfg_file)

# Загрузим веса
m.load_weights(weight_file)

# Загрузим имена классов
class_names = load_class_names(namesfile)

def boxes_iou(box1, box2):
  
    # Расчет длины и высоты каждой ограничивающей области
    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]
    
    # Расчет площади каждой ограничивающей области
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    
    # Нахождение вертикальных границ объединения (union) двух областей
    mx = min(box1[0] - width_box1/2.0, box2[0] - width_box2/2.0)
    Mx = max(box1[0] + width_box1/2.0, box2[0] + width_box2/2.0)
    
    # Нахождение ширины обединения (union) двух областей
    union_width = Mx - mx
    
    # Нахождение горизонтальных границ объединения (union) двух областей
    my = min(box1[1] - height_box1/2.0, box2[1] - height_box2/2.0)
    My = max(box1[1] + height_box1/2.0, box2[1] + height_box2/2.0)    
    
    # Нахождение высоты обединения (union) двух областей
    union_height = My - my
    
    # Расчет ширины и высоты пересечния (intersection) двух областей
    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height
   
    # Если области не пересекаются IOU равно 0
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # Расчет площади intersection oдвух областей
    intersection_area = intersection_width * intersection_height
    
    # Расчет площади union двух областей
    union_area = area_box1 + area_box2 - intersection_area
    
    # Расчет IOU
    iou = intersection_area/union_area
    
    return iou

def nms(boxes, iou_thresh):
    
    # Если нет найденных областей, то ничего не делаем
    if len(boxes) == 0:
        return boxes
    
    # Создаем тензор для отслеживания всех процентов уверенности
    det_confs = torch.zeros(len(boxes))
    
    # Уровень уверенности в каждой предсказанной области
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]

    # Сортировка индексов областей по убыванию уровня уверенности.
    _,sortIds = torch.sort(det_confs, descending = True)
    
    # Создаем пустой лист для хранения лучших областей после
    # Non-Maximal Suppression (NMS)
    best_boxes = []
    
    # Проводим Non-Maximal Suppression 
    for i in range(len(boxes)):
        
        # Сначала находим область с наивысшим уровнем уверенности
        box_i = boxes[sortIds[i]]
        
        # Проверяем, что она не 0
        if box_i[4] > 0:
            
            # Сохраняем область
            best_boxes.append(box_i)
            
            # Перебираем остальные области в списке и проводим IOU с
            # учетом предыдущей выбранной области box_i. 
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                
                # Если IOU области box_i и box_j выше, чем IOU threshold, то
                # уровень уверенности box_j равен 0. 
                if boxes_iou(box_i, box_j) > iou_thresh:
                    box_j[4] = 0
                    
    return best_boxes

def detect_objects(model, img, iou_thresh, nms_thresh):
    
    # Начало отсчета времени, для определения сколько времени занимает предсказание
    start = time.time()
    
    # Установка модели в evaluation mode.
    model.eval()
    
    # Обрезаем кратинку при необходимости до размеров входного слоя сети
    img = crop_center(img, 512, 512)
    
    # Преобразование изображения в RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Преобразование изображения из NumPy ndarray в PyTorch Tensor корректной формы
    img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    
    # Получаем предсказание модели, передавая также параметр NMS threshold.
    # для отсеивания предсказаний с маленьким уровнем уверенности
    list_boxes = model(img, nms_thresh)
    
    # Создадим новый лист для удобства работы в дальнейшем
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
    
    # Сохраняем только те области у которых IOU выше заданного порога
    boxes = nms(boxes, iou_thresh)
    
    # Останавливаем отсчет времени 
    finish = time.time()
    
    # Выведем в консоль время, затраченное на предсказание
    print('\n\nIt took {:.3f}'.format(finish - start), 'seconds to detect the objects in the image.\n')
    
    # Возвращаем полученный список
    return boxes

def print_objects(img, boxes, class_names):  
    
    # Создадим переменные для хранения информации 
    numb_obj = {}
    area_dict = {}
    
    # Посчитаем площадь изображения до преобразования
    full_width = img.shape[1]
    full_height = img.shape[0]
    com_area = full_width*full_height
    
    # Обрежем изображения до размеров входного слоя нейронной сети,
    # для того чтобы корректно отобразить получившиеся области
    resized_image = crop_center(img, 512, 512) 
    width = resized_image.shape[1]
    height = resized_image.shape[0]

    # Перебираем все элементы, полученные в результате предсказаний
    for i in range(len(boxes)):
        box = boxes[i]
        
        # Поскольку изображение выводится в первоначальном виде, необходиму посчитать 
        # дельты для корректного отображения областей
        delt_x = (full_width - width)//2
        delt_y = (full_height - height)//2
        
        # Задаем координаты областей
        x1 = int(np.around((box[0] - box[2]/2.0) * width + delt_x))
        y1 = int(np.around((box[1] - box[3]/2.0) * height + delt_y))
        x2 = int(np.around((box[0] + box[2]/2.0) * width + delt_x))
        y2 = int(np.around((box[1] + box[3]/2.0) * height + delt_y))
        
        # Считаем площадь области
        box_area = (x2 - x1)*(y2 - y1)
        box_area -= box_area*0.1
        
        # Занесем информацию по всем найденным обекта в раннее созданные словари
        if len(box) >= 7 and class_names:
            #cls_conf = box[5]
            cls_id = box[6]
            
            # Количество элементов класса
            if class_names[cls_id] in numb_obj:
                numb_obj[class_names[cls_id]] += 1
            else:
                numb_obj[class_names[cls_id]] = 1
                
            # Площадь анимаемая классом
            if class_names[cls_id] in area_dict:
                area_dict[class_names[cls_id]] += box_area
            else:
                area_dict[class_names[cls_id]] = box_area
                  
    # Сохраним информацию по количеству
    for key, value in numb_obj.items():
        result = f'Было найдено {value} растения, относящихся к классу {key}\n'
        
    # Сохраним информацию по площади в процентом соотношении
    for key, value in area_dict.items():
        percent = value/com_area * 100
        result += f'На изображении {percent:.2f}% площади это - {key}\n'
        
    # В случае, если модель не нашла объекты на изображении
    if 'result' in locals(): result += 'Найденные объекты и уровень уверенности модели:'
    else: result = 'Ничего не найдено :('
    
    return result

# Функция для выделения центральной части изображеня, заданных размеров (512х512)
def crop_center(img, crop_width = 512, crop_height = 512):
    img_shape = img.shape
    return img[(img_shape[0] - crop_width) // 2 : (img_shape[0] + crop_width) // 2,
               (img_shape[1] - crop_height) // 2 : (img_shape[1] + crop_height) // 2]   

def plot_boxes(img, boxes, class_names, plot_labels, color = None):
    
    # Определяем тензор цветов
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    
    # Определяем функция для установки цвета области
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))
        
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        
        return int(r * 255)
    
    # Определяем размеры полного изображения
    full_width = img.shape[1]
    full_height = img.shape[0]
    
    # Обрезаем значальное изображения для расчета корректного расположения облатси  
    resized_image = crop_center(img, 512, 512) 
    width = resized_image.shape[1]
    height = resized_image.shape[0]
    
    # Преобразуем изображение в RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Создаем поле для графика и настраиваем его
    plt.figure(figsize=(10,10))
    fig, a = plt.subplots(1,1)
    plt.axis('off')
    a.imshow(img)
    
    # Перебираем все полученные предсказания
    for i in range(len(boxes)):
        box = boxes[i]
        
        # Поскольку изображение выводится в первоначальном виде, необходиму посчитать 
        # дельты для корректного отображения областей
        delt_x = (full_width - width)//2
        delt_y = (full_height - height)//2
        
        # Задаем координаты областей
        x1 = int(np.around((box[0] - box[2]/2.0) * width + delt_x))
        y1 = int(np.around((box[1] - box[3]/2.0) * height + delt_y))
        x2 = int(np.around((box[0] + box[2]/2.0) * width + delt_x))
        y2 = int(np.around((box[1] + box[3]/2.0) * height + delt_y))
        
        # Проверка алгоритма
        #print(f'x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}')
        
        # Устанавливаем rgb по умолчанию на красный
        rgb = (1, 0, 0)
            
        # Используем одинаковые цвета для объектов из одного класса
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes) / 255
            green = get_color(1, offset, classes) / 255
            blue  = get_color(0, offset, classes) / 255
            
            # Если цвет не задан
            if color is None:
                rgb = (red, green, blue)
            else:
                rgb = color
        
        # Расчет высоты и длины области для ее отображения
        width_x = x2 - x1
        width_y = y1 - y2
        
        # Определения прямоугольника для обозначения области
        rect = patches.Rectangle((x1, y2),
                                 width_x, width_y,
                                 linewidth = 2,
                                 edgecolor = rgb,
                                 facecolor = 'none')

        # Отображаем полученный прямоугольник
        a.add_patch(rect)
        
        # Если plot_labels = True тогда отображаем соответсвующее обозначение
        if plot_labels:
            # Строка с названием класса и вероятностью
            conf_tx = class_names[cls_id] + ': {:.1f}'.format(cls_conf)
            
            # Отступы для текста
            lxc = (img.shape[1] * 0.266) / 100
            lyc = (img.shape[0] * 1.180) / 100
            
            # Отображение текста
            a.text(x1 + lxc, y1 - lyc, conf_tx, fontsize = 24, color = 'k',
                   bbox = dict(facecolor = rgb, edgecolor = rgb, alpha = 0.8))     
               
    # Возвращаем полученный график
    return fig
    
def detection(path,iou_thresh=0.4,nms_thresh=0.6):

    '''
    input path of any image will return detection in given image
    '''    
    # Размер изображения
    plt.rcParams['figure.figsize'] = [20.0, 10.0]

    # Загружаем изображение
    img = cv2.imread(path)

    # Устанавливаем IOU threshold. Значение по умолчанию 0.4
    iou_thresh = iou_thresh

    # Устанавливаем NMS threshold. Значение по умолчанию 0.6
    nms_thresh = nms_thresh

    # Нахождение объектов на изображении
    boxes = detect_objects(m, img, iou_thresh, nms_thresh)

    # Возвращаем полученное изображение и подпись к нему
    return (plot_boxes(img, boxes, class_names, plot_labels = True), 
            print_objects(img, boxes, class_names))