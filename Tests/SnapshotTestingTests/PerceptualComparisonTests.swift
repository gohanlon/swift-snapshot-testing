#if os(iOS) || os(tvOS) || os(macOS)
  import Accelerate.vImage
  import CoreImage
  import Metal
  import XCTest

  @testable import SnapshotTesting

  final class PerceptualComparisonTests: XCTestCase {
    // A white square, optionally with its top-left quarter in another grey.
    private func square(quarter: CGFloat = 1) -> CIImage {
      let context = CGContext(
        data: nil, width: 100, height: 100, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      )!
      context.setFillColor(CGColor(colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, components: [1, 1, 1, 1])!)
      context.fill(CGRect(x: 0, y: 0, width: 100, height: 100))
      context.setFillColor(
        CGColor(
          colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
          components: [quarter, quarter, quarter, 1]
        )!
      )
      context.fill(CGRect(x: 0, y: 50, width: 50, height: 50))
      return CIImage(cgImage: context.makeImage()!)
    }

    /// The paths a comparison can take on this machine: the CPU's always, Metal's where it runs.
    private var paths: [Bool] {
      ThresholdImageProcessorKernel.isSupported ? [true, false] : [false]
    }

    private struct RendererFailed: Error, LocalizedError {
      var errorDescription: String? { "the GPU refused the command buffer" }
    }

    private let failing = PerceptualRenderer { _, _, _, _ in throw RendererFailed() }

    func testRenderedComparisonGivesTheSameVerdicts() throws {
      for metal in paths {
        // Identical images match.
        XCTAssertNil(
          perceptuallyCompare(
            square(), square(), pixelPrecision: 0.99, perceptualPrecision: 0.98,
            renderer: .live, metal: metal
          ),
          "metal: \(metal)"
        )
        // Images that differ by less than the perceptual precision match.
        XCTAssertNil(
          perceptuallyCompare(
            square(), square(quarter: 254 / 255), pixelPrecision: 0.99,
            perceptualPrecision: 0.98, renderer: .live, metal: metal
          ),
          "metal: \(metal)"
        )
        // Images that differ by more do not, a quarter of their pixels apart.
        let message = try XCTUnwrap(
          perceptuallyCompare(
            square(), square(quarter: 0), pixelPrecision: 0.99, perceptualPrecision: 0.98,
            renderer: .live, metal: metal
          ),
          "metal: \(metal)"
        )
        let lines = message.split(separator: "\n")
        XCTAssertEqual(lines.count, 2, "metal: \(metal)")
        XCTAssertEqual(
          lines.first, "The percentage of pixels that match 0.75 is less than required 0.99",
          "metal: \(metal)"
        )
        // Black against white is a delta E of about 100, read back through half floats.
        XCTAssertEqual(
          lines.last?.hasPrefix("The lowest perceptual color precision 0.0"), true,
          "metal: \(metal)"
        )
        XCTAssertEqual(
          lines.last?.hasSuffix(" is less than required 0.98"), true, "metal: \(metal)"
        )
      }
    }

    /// The render that reports its errors writes the very bits the one that reports none wrote.
    func testRenderDestinationWritesWhatRenderToBitmapWrote() throws {
      let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
      let delta = square().applyingLabDeltaE(square(quarter: 0.3))
      let images = [
        delta,
        delta.applyingAreaAverage(),
        delta.applyingAreaMaximum(),
      ]
      for image in images {
        var legacy = try vImage_Buffer(
          width: Int(image.extent.width), height: Int(image.extent.height), bitsPerPixel: 16)
        defer { legacy.free() }
        context.render(
          image,
          toBitmap: legacy.data,
          rowBytes: legacy.rowBytes,
          bounds: image.extent,
          format: .Rh,
          colorSpace: nil
        )
        var legacy32 = try vImage_Buffer(
          width: Int(legacy.width), height: Int(legacy.height), bitsPerPixel: 32)
        defer { legacy32.free() }
        XCTAssertEqual(vImageConvert_Planar16FtoPlanarF(&legacy, &legacy32, 0), kvImageNoError)

        let rendered = try XCTUnwrap(try image.render(in: context, renderer: .live))
        defer { rendered.free() }
        XCTAssertEqual(rendered.width, legacy32.width)
        XCTAssertEqual(rendered.height, legacy32.height)
        XCTAssertEqual(rendered.rowBytes, legacy32.rowBytes)
        var row = 0
        while row < Int(rendered.height) {
          defer { row += 1 }
          let offset = row * rendered.rowBytes
          XCTAssertEqual(
            memcmp(
              rendered.data + offset, legacy32.data + offset,
              Int(rendered.width) * MemoryLayout<Float>.stride
            ),
            0,
            "row \(row) of \(image.extent)"
          )
        }
      }
    }

    func testARenderThatFailsIsNeverAMatch() throws {
      for metal in paths {
        // Images that differ, and images that do not: neither can be judged without the render.
        for new in [square(quarter: 0), square()] {
          let message = try XCTUnwrap(
            perceptuallyCompare(
              square(), new, pixelPrecision: 0.99, perceptualPrecision: 0.98,
              renderer: failing, metal: metal
            ),
            "metal: \(metal)"
          )
          XCTAssertEqual(
            message,
            """
            Newly-taken snapshot could not be compared with reference: \
            CoreImage failed to render the comparison (the GPU refused the command buffer).
            """,
            "metal: \(metal)"
          )
        }
      }
    }

    func testARenderThatFailsOnlyForTheMaximumIsNeverAMatch() throws {
      guard ThresholdImageProcessorKernel.isSupported else {
        throw XCTSkip("Metal's path does not run here.")
      }
      // The area average renders, the area maximum (asked for only once the pixels are found to
      // differ) does not.
      var renders = 0
      let failingSecond = PerceptualRenderer { image, context, bounds, destination in
        renders += 1
        if renders > 1 { throw RendererFailed() }
        try PerceptualRenderer.live.render(image, context, bounds, destination)
      }
      let message = try XCTUnwrap(
        perceptuallyCompare(
          square(), square(quarter: 0), pixelPrecision: 0.99, perceptualPrecision: 0.98,
          renderer: failingSecond, metal: true
        )
      )
      XCTAssertEqual(renders, 2)
      XCTAssertTrue(
        message.hasPrefix("Newly-taken snapshot could not be compared with reference:"), message)
    }
  }
#endif
