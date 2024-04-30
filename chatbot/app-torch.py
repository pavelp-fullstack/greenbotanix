import base64
import json
import os
import logging
import uuid
import asyncio
import torch
from torchvision import transforms, models
from PIL import Image
import os
import uuid


from aiogram import Bot, types, Dispatcher

import requests
import dotenv

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# Telegram API setup
TELEGRAM_API_KEY = os.getenv('GB_CHATBOT_TOKEN')

# Plant.id API setup for version 3
PLANT_ID_API_KEY = os.getenv('GB_PLANTID_TOKEN')
PLANT_ID_URL = 'https://plant.id/api/v3/identification'

# Image store dir
IMAGE_STORE_DIR = os.path.abspath(os.getenv('GB_IMPORTED_DOCS_DIR'))

bot = Bot(token=TELEGRAM_API_KEY)
dp = Dispatcher(bot)

model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = ["class1", "class2", "class3", "..."]

@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    intro_message = (
        "Добро пожаловать в бота-идентификатор растений! 🌱\n"
        "Я могу помочь вам определить растения по изображениям. Вот как мной пользоваться:\n\n"
        "1. Просто отправьте мне изображение растения.\n"
        "2. Подождите несколько секунд.\n"
        "3. Я скажу вам, какое это, вероятно, растение!\n\n"
        "Пожалуйста, убедитесь, что изображение четкое и растение хорошо видно для лучших результатов."
    )
    await message.answer(intro_message)


# def identify_plant(image_path):
#     with open(image_path, 'rb') as image_file:
#         encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
#         data = {
#             "images": [f'data:image/jpg;base64,{encoded_image}'],
#             "latitude": 55.75,
#             "longitude": 37.61,
#             "similar_images": True,
#             "classification_level": "all",
#             "health": "auto"
#         }
#
#         headers = {
#             'Api-Key': PLANT_ID_API_KEY,
#             'Content-Type': 'application/json'
#         }
#
#         try:
#             response = requests.post(PLANT_ID_URL, data=json.dumps(data), headers=headers).json()
#         except Exception as e:
#             print(e)
#
#     return response
#
#
# @dp.message_handler(content_types=types.ContentType.PHOTO)
# async def handle_photo(message: types.Message):
#     photo_path = os.path.join(IMAGE_STORE_DIR, f"{uuid.uuid4()}.jpg")
#     await message.photo[-1].download(photo_path)
#     response = identify_plant(photo_path)
#
#     try:
#         plant_details = response['result']['classification']['suggestions'][0]
#         name = plant_details['name']
#         probability = plant_details['probability']
#         is_healthy = response['result']['is_healthy']['binary']
#         reply_text = f"Название растения: {name}\nНадежность определения:{probability}\nРастение здорово?: {is_healthy}"
#     except (IndexError, KeyError):
#         reply_text = "Извините, мне не удалось идентифицировать растение. Попробуйте отправить другое изображение?"
#
#     await message.answer(reply_text)

@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    photo_path = os.path.join(IMAGE_STORE_DIR, f"{uuid.uuid4()}.jpg")
    await message.photo[-1].download(photo_path)

    # Load image
    image = Image.open(photo_path)
    # Apply transformations
    image = transform(image).unsqueeze(0)
    # Make an inference
    output = model(image)
    # Take the prediction with the highest score
    _, predicted = torch.max(output, 1)
    
    # TODO: convert predicted to the human-friendly description
    prediction = ""

    # prepare response based on the prediction
    if prediction.endswith('_unhealthy'):
        health_status = 'NO'
        plant_name = prediction.rsplit('_', 1)[0]
    else:
        health_status = 'YES'
        plant_name = prediction

    reply_text = f"Plant Name: {plant_name}\nAccuracy:{output.max()}\nIs Plant Healthy?: {health_status}"
    await message.answer(reply_text)

async def main():
    dp.message_handler(start, commands='start')
    dp.message_handler(handle_photo, content_types=types.ContentType.PHOTO)
    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())
