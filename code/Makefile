.PHONY: run count-loc

run:
	python3 main.py

count-loc:
	git ls-files | xargs cat | wc -l

remove-generated-data:
	cd generated_data && ls | grep -v path.py | xargs rm -rf
