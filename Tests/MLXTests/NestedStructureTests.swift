// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for `NestedItem` and `NestedDictionary` in `Nested.swift`.
class NestedStructureTests: XCTestCase {

    typealias Item = NestedItem<String, Int>

    /// A small structure: a value, an array and a dictionary.
    private func sample() -> NestedDictionary<String, Int> {
        var d = NestedDictionary<String, Int>()
        d["a"] = .value(3)
        d["b"] = .array([.value(1), .value(2)])
        d["c"] = .dictionary(["x": .value(4)])
        return d
    }

    // MARK: - access

    func testInitializersAndAccessors() {
        let empty = NestedDictionary<String, Int>()
        XCTAssertTrue(empty.isEmpty)
        XCTAssertEqual(empty.count, 0)

        let fromValues = NestedDictionary<String, Int>(values: ["a": .value(1)])
        XCTAssertEqual(fromValues.count, 1)
        XCTAssertEqual(fromValues["a"], .value(1))

        let fromItem = NestedDictionary<String, Int>(item: .dictionary(["a": .value(1)]))
        XCTAssertEqual(fromItem, fromValues)
        XCTAssertEqual(fromItem.asItem(), .dictionary(["a": .value(1)]))

        let d = sample()
        XCTAssertFalse(d.isEmpty)
        XCTAssertEqual(d.count, 3)
        XCTAssertEqual(Set(d.keys), ["a", "b", "c"])
        XCTAssertEqual(d.values.count, 3)
        XCTAssertNil(d["missing"])
    }

    func testUnwrappingSubscript() {
        var d = sample()

        // read
        XCTAssertEqual(d[unwrapping: "a"], 3)
        XCTAssertNil(d[unwrapping: "b"])  // an array, not a value
        XCTAssertNil(d[unwrapping: "missing"])

        // write a value
        d[unwrapping: "z"] = 9
        XCTAssertEqual(d["z"], .value(9))
        XCTAssertEqual(d.count, 4)

        // write nil removes the key
        d[unwrapping: "z"] = nil
        XCTAssertNil(d["z"])
        XCTAssertEqual(d.count, 3)
    }

    func testCollectionIndices() {
        let d = sample()
        var keys = [String]()
        var index = d.startIndex
        while index != d.endIndex {
            keys.append(d[index].key)
            index = d.index(after: index)
        }
        XCTAssertEqual(keys.sorted(), ["a", "b", "c"])
    }

    func testUnwrapAndAsDictionary() {
        XCTAssertNil(Item.none.unwrap())
        XCTAssertEqual(Item.value(5).unwrap() as? Int, 5)

        let array = Item.array([.value(1), .none]).unwrap() as? [Any?]
        XCTAssertEqual(array?.count, 2)
        XCTAssertEqual(array?[0] as? Int, 1)
        XCTAssertNil(array?[1] ?? nil)

        let dictionary = Item.dictionary(["k": .value(7)]).unwrap() as? [String: Any?]
        XCTAssertEqual(dictionary?["k"] as? Int, 7)

        let d = sample().asDictionary()
        XCTAssertEqual(d.count, 3)
        XCTAssertEqual((d["a"] ?? nil) as? Int, 3)
        let b = (d["b"] ?? nil) as? [Any?]
        XCTAssertEqual(b?.compactMap { $0 as? Int }, [1, 2])
        let c = (d["c"] ?? nil) as? [String: Any?]
        XCTAssertEqual((c?["x"] ?? nil) as? Int, 4)
    }

    // MARK: - description

    func testDescription() {
        XCTAssertEqual(Item.none.description, "none")
        XCTAssertEqual(Item.value(5).description, "5")
        XCTAssertEqual(Item.array([.value(1), .value(2)]).description, "[\n  1,\n  2\n]")
        XCTAssertEqual(
            Item.dictionary(["b": .value(2), "a": .array([.value(1)])]).description,
            "[\n  a: [\n    1\n  ],\n  b: 2\n]")

        var d = NestedDictionary<String, Int>()
        d["k"] = .value(1)
        XCTAssertEqual(d.description, "[\n  k: 1\n]")

        // the helper uses the indented form when the value has one
        XCTAssertEqual(indentedDescription(5, 4), "5")
        XCTAssertEqual(indentedDescription(Item.array([.value(1)]), 2), "[\n    1\n  ]")
    }

    // MARK: - map and reduce

    func testMapValuesWithPath() {
        let item: Item = .dictionary([
            "b": .array([.value(1), .value(2)]),
            "a": .value(3),
            "c": .none,
        ])

        let noPrefix = item.mapValues { (path: String, value: Int) -> String in
            "\(path)=\(value)"
        }
        let expected: NestedItem<String, String> = .dictionary([
            "a": .value("a=3"),
            "b": .array([.value("b.0=1"), .value("b.1=2")]),
            "c": .none,
        ])
        XCTAssertEqual(noPrefix, expected)

        let withPrefix = item.mapValues(prefix: "p") { (path: String, value: Int) -> String in
            path
        }
        XCTAssertEqual(withPrefix.flattenedValues(), ["p.a", "p.b.0", "p.b.1"])

        // a single value has an empty path
        XCTAssertEqual(Item.value(1).mapValues { (path: String, _: Int) in path }, .value(""))

        // the dictionary form
        let d = sample().mapValues { (path: String, value: Int) -> String in
            "\(path)=\(value)"
        }
        XCTAssertEqual(d.flattenedValues(), ["a=3", "b.0=1", "b.1=2", "c.x=4"])
    }

    func testReduce() {
        XCTAssertEqual(sample().reduce(0) { (sum: Int, v: Int) -> Int in sum + v }, 10)
        XCTAssertEqual(Item.none.reduce(5) { (sum: Int, v: Int) -> Int in sum + v }, 5)
        XCTAssertEqual(
            Item.array([.value(2), .value(3)]).reduce(1) { (p: Int, v: Int) -> Int in p * v }, 6)
    }

    func testCompactMapValuesRemovesEmptyContainers() {
        let keepEven = { (v: Int) -> Int? in v % 2 == 0 ? v : nil }

        XCTAssertEqual(Item.none.compactMapValues(keepEven), .none)
        XCTAssertEqual(Item.value(3).compactMapValues(keepEven), .none)
        XCTAssertEqual(Item.value(4).compactMapValues(keepEven), .value(4))

        // all values removed: the container becomes .none
        XCTAssertEqual(Item.array([.value(1), .value(3)]).compactMapValues(keepEven), .none)
        XCTAssertEqual(Item.dictionary(["a": .value(1)]).compactMapValues(keepEven), .none)

        // some values kept
        XCTAssertEqual(
            Item.array([.value(1), .value(2)]).compactMapValues(keepEven), .array([.value(2)]))

        let d = sample().compactMapValues(transform: keepEven)
        XCTAssertEqual(d.count, 2)
        XCTAssertEqual(d["b"], .array([.value(2)]))
        XCTAssertEqual(d["c"], .dictionary(["x": .value(4)]))
    }

    func testMapValuesTwoItems() {
        let transform = { (e1: Int, e2: Int?) -> (Int, Int?) in
            (e1 + (e2 ?? 0), e2.map { $0 * 2 })
        }

        let a: Item = .dictionary([
            "x": .value(1),
            "y": .array([.value(2), .value(3)]),
            "z": .dictionary(["w": .value(4)]),
        ])
        let b: Item = .dictionary([
            "x": .value(10),
            "y": .array([.value(20)]),
        ])

        let (r1, r2) = a.mapValues(b, transform)
        XCTAssertEqual(
            r1,
            .dictionary([
                "x": .value(11),
                "y": .array([.value(22), .value(3)]),
                "z": .dictionary(["w": .value(4)]),
            ]))
        XCTAssertEqual(
            r2,
            .dictionary([
                "x": .value(20),
                "y": .array([.value(40), .none]),
                "z": .dictionary(["w": .none]),
            ]))

        // an array against .none
        let (s1, s2) = Item.array([.value(1)]).mapValues(Item.none, transform)
        XCTAssertEqual(s1, .array([.value(1)]))
        XCTAssertEqual(s2, .array([.none]))

        // .none against anything
        let (t1, t2) = Item.none.mapValues(Item.value(1), transform)
        XCTAssertEqual(t1, .none)
        XCTAssertEqual(t2, .none)
    }

    func testMapValuesThreeItems() {
        func run(_ x: Item, _ y: Item, _ z: Item) -> (Item, Item, Item) {
            x.mapValues(y, z) { (a: Int, b: Int?, c: Int?) -> (Int, Int?, Int?) in
                (a + (b ?? 0) + (c ?? 0), b, c)
            }
        }
        func assertTriple(
            _ r: (Item, Item, Item), _ e: (Item, Item, Item), line: UInt = #line
        ) {
            XCTAssertEqual(r.0, e.0, line: line)
            XCTAssertEqual(r.1, e.1, line: line)
            XCTAssertEqual(r.2, e.2, line: line)
        }

        // .none receiver
        assertTriple(run(.none, .value(1), .value(2)), (.none, .none, .none))

        // values
        assertTriple(run(.value(1), .none, .none), (.value(1), .none, .none))
        assertTriple(run(.value(1), .value(2), .none), (.value(3), .value(2), .none))
        assertTriple(run(.value(1), .none, .value(3)), (.value(4), .none, .value(3)))
        assertTriple(run(.value(1), .value(2), .value(3)), (.value(6), .value(2), .value(3)))

        // arrays (the second and third may be shorter)
        let a1: Item = .array([.value(1), .value(2)])
        let a2: Item = .array([.value(10)])
        let a3: Item = .array([.value(5), .value(6)])
        assertTriple(
            run(a1, .none, .none),
            (.array([.value(1), .value(2)]), .array([.none, .none]), .array([.none, .none])))
        assertTriple(
            run(a1, a2, .none),
            (
                .array([.value(11), .value(2)]), .array([.value(10), .none]),
                .array([.none, .none])
            ))
        assertTriple(
            run(a1, .none, a3),
            (.array([.value(6), .value(8)]), .array([.none, .none]), a3))
        assertTriple(
            run(a1, a2, a3),
            (.array([.value(16), .value(8)]), .array([.value(10), .none]), a3))

        // dictionaries (the second and third may miss keys)
        let d1: Item = .dictionary(["p": .value(1), "q": .value(2)])
        let d2: Item = .dictionary(["p": .value(10)])
        let d3: Item = .dictionary(["q": .value(100)])
        assertTriple(
            run(d1, .none, .none),
            (d1, .dictionary(["p": .none, "q": .none]), .dictionary(["p": .none, "q": .none])))
        assertTriple(
            run(d1, d2, .none),
            (
                .dictionary(["p": .value(11), "q": .value(2)]),
                .dictionary(["p": .value(10), "q": .none]),
                .dictionary(["p": .none, "q": .none])
            ))
        assertTriple(
            run(d1, .none, d3),
            (
                .dictionary(["p": .value(1), "q": .value(102)]),
                .dictionary(["p": .none, "q": .none]),
                .dictionary(["p": .none, "q": .value(100)])
            ))
        assertTriple(
            run(d1, d2, d3),
            (
                .dictionary(["p": .value(11), "q": .value(102)]),
                .dictionary(["p": .value(10), "q": .none]),
                .dictionary(["p": .none, "q": .value(100)])
            ))
    }

    func testDictionaryMapValuesVariants() {
        var d1 = NestedDictionary<String, Int>()
        d1["w"] = .array([.value(1), .value(2)])
        var d2 = NestedDictionary<String, Int>()
        d2["w"] = .array([.value(10), .value(20)])
        var d3 = NestedDictionary<String, Int>()
        d3["w"] = .array([.value(100)])

        // two dictionaries, one result
        let one = d1.mapValues(d2) { (a: Int, b: Int?) -> Int in a * 1000 + (b ?? 0) }
        XCTAssertEqual(one["w"], .array([.value(1010), .value(2020)]))

        // three dictionaries, two results
        let (r1, r2) = d1.mapValues(d2, d3) { (a: Int, b: Int?, c: Int?) -> (Int, Int?) in
            (a + (b ?? 0) + (c ?? 0), c)
        }
        XCTAssertEqual(r1["w"], .array([.value(111), .value(22)]))
        XCTAssertEqual(r2["w"], .array([.value(100), .none]))
    }

    // MARK: - flatten and unflatten

    func testFlattenWithPrefix() {
        let flat = sample().flattened(prefix: "root").map { "\($0.0)=\($0.1)" }
        XCTAssertEqual(flat, ["root.a=3", "root.b.0=1", "root.b.1=2", "root.c.x=4"])

        XCTAssertTrue(Item.none.flattened().isEmpty)
        XCTAssertTrue(Item.none.flattenedValues().isEmpty)
        XCTAssertEqual(Item.value(1).flattened(prefix: "v").map { $0.0 }, ["v"])
    }

    func testUnflattenForms() {
        // empty input gives an empty dictionary
        XCTAssertEqual(Item.unflattened([(String, Int)]()), .dictionary([:]))

        // integer keys make an array, missing indices are .none
        XCTAssertEqual(
            Item.unflattened([("0", 1), ("2", 3)]), .array([.value(1), .none, .value(3)]))

        // the [String: Element] form
        let d = NestedDictionary<String, Int>.unflattened(["a.b": 1, "a.c": 2, "d": 3])
        XCTAssertEqual(d.count, 2)
        XCTAssertEqual(d["a"], .dictionary(["b": .value(1), "c": .value(2)]))
        XCTAssertEqual(d["d"], .value(3))

        // round trip
        let s = sample()
        XCTAssertEqual(NestedDictionary<String, Int>.unflattened(s.flattened()), s)
    }
}
