import os

FILE_EXISTS_ERROR = (17, 'File exists')


def create_folder(path):
    """
    Checks if the path exists, if not creates it.
    :param path: A valid path that might not exist
    :return: An indication if the folder was created
    """
    folder_missing = not os.path.exists(path)

    if folder_missing:
        # Using makedirs since the path hierarchy might not fully exist.
        try:
            os.makedirs(path)
        except OSError as e:
            if (e.errno, e.strerror) == FILE_EXISTS_ERROR:
                print(e)
            else:
                raise

        print('Created folder {0}'.format(path))

    return folder_missing
