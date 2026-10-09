"""DocTags (what a Docling-family reader model writes) become Markdown with the real docling-core."""

from __future__ import annotations

import unittest

from PIL import Image

from rag_search.core.conversion import vlm_worker


class DocTagsTests(unittest.TestCase):
    def test_doctags_become_markdown_with_a_table(self):
        tags = ("<doctag><section_header_level_1><loc_5><loc_5><loc_50><loc_20>Report</section_header_level_1>"
                "<otsl><loc_5><loc_30><loc_90><loc_60><fcel>A<fcel>B<nl><fcel>1,2<fcel>3<nl></otsl></doctag><|end_of_text|>")
        md = vlm_worker.doctags_to_markdown(tags, Image.new("RGB", (100, 100), "white"))
        self.assertIn("## Report", md)
        self.assertIn("| 1,2", md)


if __name__ == "__main__":
    unittest.main()
