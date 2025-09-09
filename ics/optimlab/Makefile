grade: grader
	./grader

grader: main.c poly.c measure_time_std.o
	gcc $^ -o $@ -O2

handin: poly.c
	$(eval USERNAME := $(shell whoami))
	$(eval ZIPNAME := $(shell echo '$(USERNAME)-lab5-handin.zip'))
	zip $(ZIPNAME) poly.c

clean:
	rm grader