import pandas as pd

import os
import asyncio
import aiohttp  # pip install aiohttp
import aiofiles  # pip install aiofiles

REPORTS_FOLDER = "data"
FILES_PATH = os.path.join(REPORTS_FOLDER, "images")


def download_files_from_report(urls):
    os.makedirs(FILES_PATH, exist_ok=True)
    sema = asyncio.BoundedSemaphore(5)

    async def fetch_file(url):
        fname = url.split("/")[-1]
        async with sema, aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                assert resp.status == 200
                data = await resp.read()

        async with aiofiles.open(
            os.path.join(FILES_PATH, fname), "wb"
        ) as outfile:
            await outfile.write(data)

    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(fetch_file(url)) for url in urls]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


df  = pd.read_csv("sample_train.csv")
urls = df["url"]

download_files_from_report(urls=urls)

