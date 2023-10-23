"""
I wrote this async version which should be faster, but of course that 
sending 1000 requests implies in a lot of errors. So, although I does 
run faster, I'm not really sure if its error safe. It will work just 
as the sync version if every page is returned by the jikan API.
"""
import aiohttp
import json
import asyncio
from aiofiles import open as aio_open

async def fetch_anime_data(session, page):
    try:
        async with session.get(f"https://api.jikan.moe/v4/anime?page={page}") as response:
            result = await response.json()
            print(f"Requesting page: {page}")

            async with aio_open("/home/gabriel-dornelles/Documents/gabriel/anime_recommender/dataset.json", 'a') as outfile:
                for item in result["data"]:
                    buffer = {item["title"]: item}
                    await outfile.write(json.dumps(buffer, indent=4) + ',\n')

    except Exception as e:
        print(f"Request failed for page {page}, trying again: {str(e)}")
        await fetch_anime_data(session, page)  # Retry the request

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_anime_data(session, i) for i in range(1, 1021)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
