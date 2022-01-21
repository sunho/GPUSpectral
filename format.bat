cd src
for /f %%f in ('dir *.cpp *.h /b/s') do clang-format -i %%f