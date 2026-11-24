import shutil
import arabic_reshaper
from bidi import get_display


def ar(text):
    """Print-only: join the letters and flip the order so the terminal shows it correctly."""
    return get_display(arabic_reshaper.reshape(text))


def ar_right(text):
    """Print-only: same as ar(), but pushed to the right edge of the terminal."""
    width = shutil.get_terminal_size().columns   # number of columns of the terminal
    return ar(text).rjust(width)                 # pad with spaces on the left


print(ar_right("احمد في المدرسة وقد كان يفترض أن يكون هنا ليلا."))