@ECHO OFF

set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set OUTPUTDIR=..
set DOCTREEDIR=../.doctrees

%SPHINXBUILD% -b html -d %DOCTREEDIR% %SOURCEDIR% %OUTPUTDIR%
copy /Y %SOURCEDIR%\_static\custom.css %OUTPUTDIR%\_static\custom.css
