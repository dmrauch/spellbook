*****
macOS
*****


Creating a Screencast As an Animated GIF
========================================

- Press :kbd:`Command ⌘` + :kbd:`Shift ⇧` + :kbd:`5` to bring up the screenshot
  / screen recording menu

- Select "Record Entire Screen" or "Record Selected Portion" and start the
  recording ("Stop" will be shown button on the touchbar)

- The recording yields a ``.mov``-file that has to be converted to a ``.gif``.
  This can be done with *ffmpeg* and *gifsicle* [#macOSScreencastGIF]_::

     $ ffmpeg -i <infile>.mov -s 600x400 -pix_fmt rgb24 -r 10 -f gif - \
           | gifsicle --optimize=3 --delay=5 > <outfile>.gif


.. rubric:: Links & References

.. [#macOSScreencastGIF] https://gist.github.com/dergachev/4627207
