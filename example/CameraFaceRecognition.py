from facelib import add_from_folder
from facelib import WebcamVerify
add_from_folder(folder_path='./feng/', person_name='feng')
add_from_folder(folder_path='./qiu/', person_name='qiu')

verifier = WebcamVerify(update=True)
verifier.run()