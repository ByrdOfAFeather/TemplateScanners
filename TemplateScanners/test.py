import templatescannersbeta
import unittest
import os

BASE_DIR = "testfiles"


def build_template_dict():
	template_names = os.listdir(f"{BASE_DIR}/templates")
	template_dict = {}
	for template_name in template_names:
		if template_name.startswith("_"): continue
		template_dict[template_name] = [f"{BASE_DIR}/templates/{template_name}/{path}" for path in os.listdir(f"{BASE_DIR}/templates/{template_name}")]
	return template_dict


def video_scanner_test():
	template_dict = build_template_dict() 
	scanner = templatescannersbeta.ThreadedVideoScan() 
	results_non_adaptive = scanner.run(template_dict, f"{BASE_DIR}/videos/smalltest.mp4", .5)
	results_adaptive = scanner.run_adaptive_threshold(template_dict, f"{BASE_DIR}/videos/smalltest.mp4")

	print(f"THESE ARE THE RESULTS I GOT FROM NON-ADAPTIVE {results_non_adaptive}")
	print(f"THESE ARE THE RESULTS I GOT FROM ADAPTIVE {results_adaptive}") 


def templatescanners_test():
	template_dict = build_template_dict()
	template_scanner = templatescannersbeta.TemplateScanner(template_dict, .1)

	test1 = template_scanner.scan(f"{BASE_DIR}/images/2.png")
	assert test1 in ["Jump", "Attack", "Move", ""], "Scan produced an impossible result!"


def main():
	print("RUNNING TESTS FOR TEMPLATE SCANNERS")
	templatescanners_test()
	print("TEMPLATE SCANNERS PASSED")
	print("RUNNING TESTS FOR VIDEO SCANNERS")
	video_scanner_test()
	print("VIDEO SCANNERS PASSED")


main() 