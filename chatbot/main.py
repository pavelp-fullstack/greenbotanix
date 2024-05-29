from aiogram import Bot, Dispatcher, executor, types
from aiogram.types.input_file import InputFile
from config import API_TOKEN
from performing_detection import detection

import os
import logging
import uuid
import asyncio
import dotenv
import os

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

IMAGE_STORE_DIR = 'data/images/'
IMAGE_DET_DIR = 'data/detection/'

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Функцмя команды start, выводит изображение
@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    intro_message = (
        "Добро пожаловать в бота-идентификатор растений! 🌱\n"
        "Я могу помочь вам определить растения по изображениям. Вот как мной пользоваться:\n\n"
        "1. Просто отправьте мне изображение растения.\n"
        "2. Подождите несколько секунд.\n"
        "3. Я скажу вам, сорняк это или культура!\n\n"
        "Пожалуйста, убедитесь, что изображение четкое и растение хорошо видно для лучших результатов."
    )
    await message.answer(intro_message)


@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    
    # Формируем путь для сохранения отправленной пользователем фотографии
    photo_path = os.path.join(IMAGE_STORE_DIR, f"{uuid.uuid4()}.jpg")
    
    # Сохраняем фотографию
    await message.photo[-1].download(photo_path)
    
    # Получаем предсказание
    response = detection(photo_path)
    
    # Вывод текста
    reply_text = response[1]
    await message.answer(reply_text)
    
    # Если модель что-то нашла, то сохраняем фото предсказания в заданную директорию
    if reply_text != 'Ничего не найдено :(':
        photo = response[0]
        plot_path = os.path.join(IMAGE_DET_DIR, f"{uuid.uuid4()}.jpg")
        photo.savefig(plot_path, bbox_inches='tight')
        photo = InputFile(plot_path)
        # Отображаем сохраненное фото
        await message.answer_photo(photo)

# @dp.message_handler(commands=['stop']) 
# async def stop(message: types.Message): 
#     dp.stop_bot()
#     await message.answer('Bot stopped')

async def main():
    dp.message_handler(start, commands='start')
    dp.message_handler(handle_photo, content_types=types.ContentType.PHOTO)
    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())