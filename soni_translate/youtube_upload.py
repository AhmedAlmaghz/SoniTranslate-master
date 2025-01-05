from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
import os
from soni_translate.utils import get_directory_files
from soni_translate.logging_setup import logger

def upload_video_to_youtube(video_path, channel_id, token, title, description, tags):
    """
    Uploads a video to YouTube using the provided credentials.

    Args:
        video_path: Path to the video file.
        channel_id: ID of the YouTube channel.
        token: OAuth 2.0 token for authentication.
        title: Title of the video.
        description: Description of the video.
        tags: List of tags for the video.
    """

    youtube = build("youtube", "v3", credentials=token)

    try:
        request = youtube.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "categoryId": "22",
                    "title": title,
                    "description": description,
                    "tags": tags,
                },
                "status": {"privacyStatus": "private"},
            },
            media_body=MediaFileUpload(video_path, chunksize=-1, resumable=True),
        )

        response = None
        error = None
        retry = 0

        while response is None:
            try:
                logger.info("Uploading file...")
                status, response = request.next_chunk()
                if response is not None:
                    if "id" in response:
                        logger.info(
                            "Video id '%s' was successfully uploaded." % response["id"]
                        )
                        return response["id"]
                    else:
                        exit("The upload failed with an unexpected response: %s" % response)
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    error = "A retriable HTTP error %d occurred:\n%s" % (e.resp.status, e.content)
                else:
                    raise
            except Exception as e:
                error = "A retriable error occurred: %s" % e

            if error is not None:
                logger.info(error)
                retry += 1
                if retry > 5:
                    exit("No longer attempting to retry.")

                max_sleep = 2 ** retry
                sleep_seconds = random.random() * max_sleep
                logger.info("Sleeping %f seconds and then retrying..." % sleep_seconds)
                time.sleep(sleep_seconds)

    except HttpError as e:
        logger.error(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None
