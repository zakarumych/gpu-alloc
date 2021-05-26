# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Support for `ash` API.

### Fixed
- Erupt checks for correct extension to determine buffer device feature availability.

## [0.4.7] - 2021-05-22

### Fixed
- Add missing block allocation counting for free-list allocator.

## [0.4.6] - 2021-05-21

### Fixed
- Fixed freeing tail block in its region for FreeBlockAllocator.

## [0.4.5] - 2021-05-03

### Fixed
- Fixed checking region-block overlap

## [0.4.4] - 2021-04-28
### Changed
- `freelist` feature to use free-list based allocator instead of arena allocator now enabled by default

### Fixed
- Fix free-list based allocator

## [0.4.3] - 2021-04-20
### Changed
- Drop check error message is suppressed during unwinding

## [0.4.2] - 2021-03-30
###  Fixed
- Allocator type gets back Send and Sync implementations

## [0.4.1] - 2021-03-30 [YANKED]
### Added
- Free-list based allocator to replace arena allocator for better memory reuse.
  Active only when feature `freelist` is enabled.

### Fixed
- SemVer adhesion

## [0.3.1] - 2021-03-27 [YANKED]
### Fixed
- Typos in public API

## [0.3.0]
### Changed
- Mapping and unmapping now requires mutable reference,
  preventing aliasing and double-mapping problems.
- Simple heap budget checking removed as the device may report
  significantly smaller heap size.


## [0.2.0] - 2020-11-13
### Changed
- Allowed to map memory though shared memory block.

## [0.1.1] - 2020-11-10
### Added
- Graphics API agnostic general purpose gpu memory allocator.
