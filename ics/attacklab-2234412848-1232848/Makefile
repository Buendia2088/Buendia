FILES := phase1.txt phase2.txt phase3.txt phase4.txt phase5.txt ctarget rtarget cookie.txt

submit:
	@for file in $(FILES); do \
		if [ ! -f $$file ]; then \
			echo "Warning: $$file does not exist and will be skipped"; \
		fi; \
	done
	@zip ${USER}-handin.zip $(FILES)

grade:
	python3 autograder.py


