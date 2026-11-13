//
//  FlannBasedMatcherTest.swift
//

import XCTest
import OpenCV

class FlannBasedMatcherTest: OpenCVTestCase {

    // FlannBasedMatcher::write output for the default constructor (same as FlannBasedDescriptorMatcherTest.java)
    let ymlParamsDefault = "%YAML 1.2\n---\n"
        + "format: 3\n"
        + "indexParams:\n"
        + "   -\n"
        + "      name: algorithm\n"
        + "      type: 9\n"   // FLANN_INDEX_TYPE_ALGORITHM
        + "      value: 1\n"
        + "   -\n"
        + "      name: trees\n"
        + "      type: 4\n"   // FLANN_INDEX_TYPE_32S
        + "      value: 4\n"
        + "searchParams:\n"
        + "   -\n"
        + "      name: checks\n"
        + "      type: 4\n"
        + "      value: 32\n"
        + "   -\n"
        + "      name: eps\n"
        + "      type: 5\n"   // FLANN_INDEX_TYPE_32F
        + "      value: 0.\n"
        + "   -\n"
        + "      name: explore_all_trees\n"
        + "      type: 8\n"   // FLANN_INDEX_TYPE_BOOL
        + "      value: 0\n"
        + "   -\n"
        + "      name: sorted\n"
        + "      type: 8\n"
        + "      value: 1\n"

    func tempFileName(_ ext: String) -> String {
        return (NSTemporaryDirectory() as NSString).appendingPathComponent("\(UUID().uuidString).\(ext)")
    }

    func writeParams(_ matcher: FlannBasedMatcher) -> String {
        let filename = tempFileName("yml")
        defer { try? FileManager.default.removeItem(atPath: filename) }
        matcher.write(fileName: filename)
        return (try? String(contentsOfFile: filename, encoding: .utf8)) ?? ""
    }

    func testDefaultConstructorParams() {
        XCTAssertEqual(ymlParamsDefault, writeParams(FlannBasedMatcher()))
    }

    func testCustomParamsAreStored() {
        // Covers the algorithm, Int, Double, Bool (true/false) and "eps" (float) branches.
        let matcher = FlannBasedMatcher(
            indexParams: ["algorithm": 2, "branching": 16, "cb_index": 1.0, "iterations": 5],
            searchParams: ["checks": 64, "eps": 2.0, "explore_all_trees": true, "sorted": false])

        let expected = "%YAML 1.2\n---\n"
            + "format: 3\n"
            + "indexParams:\n"
            + "   -\n"
            + "      name: algorithm\n"
            + "      type: 9\n"   // FLANN_INDEX_TYPE_ALGORITHM
            + "      value: 2\n"
            + "   -\n"
            + "      name: branching\n"
            + "      type: 4\n"   // FLANN_INDEX_TYPE_32S
            + "      value: 16\n"
            + "   -\n"
            + "      name: cb_index\n"
            + "      type: 6\n"   // FLANN_INDEX_TYPE_64F
            + "      value: 1.\n"
            + "   -\n"
            + "      name: iterations\n"
            + "      type: 4\n"
            + "      value: 5\n"
            + "searchParams:\n"
            + "   -\n"
            + "      name: checks\n"
            + "      type: 4\n"
            + "      value: 64\n"
            + "   -\n"
            + "      name: eps\n"
            + "      type: 5\n"   // FLANN_INDEX_TYPE_32F
            + "      value: 2.\n"
            + "   -\n"
            + "      name: explore_all_trees\n"
            + "      type: 8\n"   // FLANN_INDEX_TYPE_BOOL
            + "      value: 1\n"
            + "   -\n"
            + "      name: sorted\n"
            + "      type: 8\n"
            + "      value: 0\n"

        XCTAssertEqual(expected, writeParams(matcher))
    }

    func testStringParamIsStored() {
        // FLANN_INDEX_SAVED (254) takes the index file name as a string parameter.
        let matcher = FlannBasedMatcher(indexParams: ["algorithm": 254, "filename": "flann_index"],
                                        searchParams: ["checks": 32])

        let yml = writeParams(matcher)
        XCTAssertTrue(yml.contains("      name: filename\n      type: 7\n      value: flann_index\n"), yml)  // FLANN_INDEX_TYPE_STRING
        XCTAssertTrue(yml.contains("      name: algorithm\n      type: 9\n      value: 254\n"), yml)
    }

    func testIndexParamsOnlyKeepsDefaultSearchParams() {
        let matcher = FlannBasedMatcher(indexParams: ["algorithm": 1, "trees": 8])

        let expected = "%YAML 1.2\n---\n"
            + "format: 3\n"
            + "indexParams:\n"
            + "   -\n"
            + "      name: algorithm\n"
            + "      type: 9\n"
            + "      value: 1\n"
            + "   -\n"
            + "      name: trees\n"
            + "      type: 4\n"
            + "      value: 8\n"
            + "searchParams:\n"
            + "   -\n"
            + "      name: checks\n"
            + "      type: 4\n"
            + "      value: 32\n"
            + "   -\n"
            + "      name: eps\n"
            + "      type: 5\n"
            + "      value: 0.\n"
            + "   -\n"
            + "      name: explore_all_trees\n"
            + "      type: 8\n"
            + "      value: 0\n"
            + "   -\n"
            + "      name: sorted\n"
            + "      type: 8\n"
            + "      value: 1\n"

        XCTAssertEqual(expected, writeParams(matcher))
    }

    func testDefaultConstructorKDTreeMatch() {
        let train = Mat(rows: 50, cols: 32, type: CvType.CV_32F)
        Core.randu(dst: train, low: 0, high: 100)

        let matcher = FlannBasedMatcher()
        var matches = [DMatch]()
        matcher.match(queryDescriptors: train, trainDescriptors: train, matches: &matches)

        XCTAssertEqual(50, matches.count)
        for (i, m) in matches.enumerated() {
            XCTAssertEqual(Int32(i), m.trainIdx)
            XCTAssertEqual(0, m.distance, accuracy: OpenCVTestCase.FEPS)
        }
    }

    func testLSHMatchOnBinaryDescriptors() {
        // Binary (CV_8U) descriptors need an LSH index; the default KD-tree rejects them,
        // so a successful match also shows that "algorithm": 6 was applied.
        let train = Mat(rows: 50, cols: 32, type: CvType.CV_8U)
        Core.randu(dst: train, low: 0, high: 256)

        let matcher = FlannBasedMatcher(
            indexParams: ["algorithm": 6, "table_number": 6, "key_size": 12, "multi_probe_level": 1],
            searchParams: ["checks": 50])
        var matches = [DMatch]()
        matcher.match(queryDescriptors: train, trainDescriptors: train, matches: &matches)

        XCTAssertEqual(50, matches.count)
        for (i, m) in matches.enumerated() {
            XCTAssertEqual(Int32(i), m.trainIdx)
            XCTAssertEqual(0, m.distance, accuracy: OpenCVTestCase.FEPS)
        }
    }
}
