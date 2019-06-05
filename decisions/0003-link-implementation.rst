===================
Link Implementation
===================

Links are how we track reuse across Bundles.

It's more intuitive for client applications to think of Links as happening
between BundleVersions, but if we delete a Snapshot, a BundleVersion

Why are Snapshots different from Bundle Versions?
* Race conditions?
* So the repo can manage the Link dependency checking?

So when doing a PATCH with link info:
* we have to include the Bundle UUID, BundleVersion number -> we derive snapshot.