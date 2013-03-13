test:
	python setup.py test

release:
	python scripts/make-release.py

upload-docs:
	$(MAKE) -C docs html dirhtml latex
	$(MAKE) -C docs/_build/latex all-pdf
	cd docs/_build/; mv html jinja-docs; zip -r jinja-docs.zip jinja-docs; mv jinja-docs html
	scp -r docs/_build/dirhtml/* pocoo.org:/var/www/jinja.pocoo.org/docs/
	scp -r docs/_build/latex/Jinja2.pdf pocoo.org:/var/www/jinja.pocoo.org/docs/jinja-docs.pdf
	scp -r docs/_build/jinja-docs.zip pocoo.org:/var/www/jinja.pocoo.org/docs/

.PHONY: test
