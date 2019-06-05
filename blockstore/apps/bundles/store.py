"""
High Level Concepts:

Snapshot and StagedDraft are representations of a set of files that are
independent of how they are stored or accessed. A StagedDraft has a reference to
the Snapshot it was based on. Snapshots are not aware of StagedDrafts.

SnapshotRepo and DraftRepo provide strorage and URLs for Snapshots and
StagedDrafts respectively. Again, DraftRepo depends on SnapshotRepo, but
SnapshotRepo is unaware of DraftRepo.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import UUID
import codecs
import hashlib
import logging
import json

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile, File
from django.dispatch import Signal

import attr

# Pylint doesn't know how to introspect attr.s() structs, and can't tell that
# Snapshot.files or StagedDraft.files_to_overwrite are in fact dicts and can do
# all of the things listed below:
# pylint: disable=unsubscriptable-object,unsupported-membership-test,not-an-iterable

logger = logging.getLogger(__name__)
snapshot_created = Signal(providing_args=["bundle_uuid", "hash_digest"])


@attr.s(frozen=True)
class FileInfo:
    """
    Low level metadata about a file in a Snapshot or Draft.

    Backend-agnostic, so no details about exactly where or how it's stored. The
    `hash_digest` exists to make it easy to see what things have changed.
    """
    path = attr.ib(type=str)
    public = attr.ib(type=bool)
    size = attr.ib(type=int)
    hash_digest = attr.ib(type=bytes)

    @staticmethod
    def generate_hash(django_file):
        """Given a Django File object, return a hash."""
        hash_obj = create_hash()
        for chunk in django_file.chunks():
            hash_obj.update(chunk)
        return hash_obj

    @classmethod
    def from_json_dict(cls, json_dict):
        """Given a JSON parsed dictionary, create path->FileInfo dict mapping."""
        file_info_dict = {}
        for pathname, file_info in json_dict.items():
            if file_info is None:
                # Used by Drafts to indicate this file should be deleted.
                file_info_dict[pathname] = None
            else:
                file_info_dict[pathname] = cls(
                    path=pathname,
                    public=file_info[0],
                    size=file_info[1],
                    hash_digest=bytes_from_hex_str(file_info[2])
                )
        return file_info_dict


@attr.s(frozen=True)
class Dependency:
    """
    A Dependency is a pointer to exactly one Bundle + Version + Snapshot.
    """
    bundle_uuid = attr.ib()
    version = attr.ib(type=int)
    snapshot_digest = attr.ib(type=bytes)

    @classmethod
    def from_json_dict(cls, json_dict):
        return cls(
            bundle_uuid=UUID(json_dict["bundle_uuid"]),
            version=json_dict["version"],
            snapshot_digest=bytes_from_hex_str(json_dict["snapshot_digest"]),
        )


@attr.s(frozen=True)
class Link:
    """
    A Link is a named Dependency + all of its transitive dependencies.

    Links are named because it's totally fine to have Links that point to
    different versions of a Bundle (e.g Bundle Av1, Bundle Av2), but we want
    some stable way of identifying and differentiating between those references.

    The indirect dependencies are a flat list. We don't try to keep track of all
    the nested dependency relations. Instead, we want the Link to answer the
    question of "If I wanted to download this entire Bundle and all the things
    it links to, what is the complete list of other BundleVersions that I have
    to download to get all the dependencies?
    """
    name = attr.ib(type=str)
    direct_dependency = attr.ib(type=Dependency)
    indirect_dependencies = attr.ib(type=list)


class LinkCollection:
    """
    This data structure has to hold all dependencies for a Bundle.

    Like most classes in this module, `LinkCollection` is immutable. Calling
    `with_updated_link()` and passing in a new `Link` will return a new
    `LinkCollection` rather than modifying the object in place.

    There are a few critical query patterns that we have to support here:

    1. Quickly return the entire set of things that needs to be downloaded. This
    is useful when we want to do an export of some sort.

    2. Determine if there is a dependency cycle. Dependency cycles break our
    assumptions and will lead us to endlessly prompt for updates, so we have to
    prevent them from happening.

    3. Add a new dependency (and all of its transitive dependencies).
    """
    def __init__(self, bundle_uuid: UUID, links: List[Link]):
        """
        Initialize with a `bundle_uuid` and list of Link objects.

        Will raise a ValueError if the data is malformed in some way (duplicate
        Link names or dependency cycles).
        """
        self._check_for_duplicates(links)
        self._check_for_cycles(bundle_uuid, links)

        self.bundle_uuid = bundle_uuid
        self.names_to_links = {link.name: link for link in links}

    def __eq__(self, other):
        return (
            self.bundle_uuid == other.bundle_uuid and
            self.names_to_links == other.names_to_links
        )

    def __getitem__(self, name):
        return self.names_to_links[name]

    def __iter__(self):
        return iter(self.names_to_links.values())

    def __bool__(self):
        return bool(self.names_to_links)

    @classmethod
    def from_json_dict(cls, bundle_uuid, json_dict):
        def _parse_dep(dep_info):
            return Dependency(
                bundle_uuid=dep_info['bundle_uuid'],
                version=dep_info['version'],
                snapshot_digest=bytes_from_hex_str(dep_info['snapshot_digest'])
            )

        links = []
        for link_name, link_info in json_dict.items():
            direct_dep = _parse_dep(link_info['direct'])
            indirect_deps = [_parse_dep(dep_info) for dep_info in link_info['indirect']]
            links.append(
                Link(link_name, direct_dep, indirect_deps)
            )

        print(links)

        return LinkCollection(bundle_uuid, links)

    def get_direct_dep(self, name, default=None):
        link = self.names_to_links.get(name)
        if link:
            return link.direct_dependency
        return None

    def _check_for_duplicates(self, links):
        seen_names = set()
        for link in links:
            if link.name in seen_names:
                raise ValueError(f"Duplicate link name not allowed: {link.name}")

    def _check_for_cycles(self, bundle_uuid, links):
        for link in links:
            if link.direct_dependency == bundle_uuid:
                raise ValueError(
                    f"A Bundle cannot have any version of itself as a dependency (see Link {link.name})."
                )
            for dep in link.indirect_dependencies:
                if dep.bundle_uuid == bundle_uuid:
                    raise ValueError(
                        f"Cycle detected: Link {link.name} requires Bundle {bundle_uuid}"
                    )

    def all_dependencies(self):
        dependencies = set(link.direct_dependency for link in self)
        for link in self:
            dependencies |= set(link.indirect_dependencies)
        return list(sorted(dependencies))

    def with_updated_link(self, link):
        new_links = list(lk for lk in self if lk.name != link.name) + [link]
        return LinkCollection(self.bundle_uuid, new_links)

    def with_updated_links(self, links):
        """Return a new LinkCollection combining self with overrides from links."""
        names_to_links = {link.name: link for link in self}
        for link in links:
            names_to_links[link.name] = link
        return LinkCollection(self.bundle_uuid, names_to_links.values())


@attr.s(frozen=True)
class Snapshot:
    """
    Represents an immutable, point-in-time representation of a file tree.

    Snapshots are only created for states that we want to permanently
    perserve, much like a commit in a version control system. We do not want to
    create a bunch of Snapshots for intermediate versions.

    This class is a low level primitive that just stores Bundle content. Most
    metadata about a Bundle should be added to or referenced from the Bundle and
    BundleVersion models.

    Snapshots are never modified after they are created. In rare cases of
    malicious or illegal content, snapshots may be deleted.
    """
    bundle_uuid = attr.ib()

    # Map of paths to FileInfo objects
    files = attr.ib(type=dict)

    # Captures our direct and indirect dependencies.
    links = attr.ib(type=LinkCollection)

    # 20-byte BLAKE2 hash
    hash_digest = attr.ib(type=bytes)

    # Datetime with UTC Timezone
    created_at = attr.ib(type=datetime)

    @classmethod
    def create(cls, bundle_uuid, files, links=None, created_at=None):
        """Create a Snapshot."""
        if links is None:
            links = LinkCollection(bundle_uuid, [])

        created_at = created_at or datetime.now(timezone.utc)
        str_to_be_hashed = json.dumps(
            [bundle_uuid, created_at, sorted(files.items()), sorted(links)],
            cls=BundleDataJSONEncoder,
            indent=None,
            separators=(',', ':'),
        )
        hash_digest = create_hash(str_to_be_hashed.encode('utf-8')).digest()
        return cls(
            bundle_uuid=bundle_uuid,
            files=files,
            links=links,
            hash_digest=hash_digest,
            created_at=created_at,
        )

    class NotFoundError(Exception):
        pass


@attr.s(frozen=True)
class StagedDraft:
    """
    StagedDraft are stored as diffs on top of existing Snapshots.
    """
    uuid = attr.ib()
    bundle_uuid = attr.ib()
    name = attr.ib(type=str)

    base_snapshot = attr.ib(type=Snapshot)  # This can be None for a new Bundle

    # A dict of all files that have changed. The keys are paths and the values
    # are FileInfo (mirroring a Snapshot's `files` attribute). A value of None
    # means that path has been deleted in the Draft.
    files_to_overwrite = attr.ib(type=dict)
    created_at = attr.ib(type=datetime)
    updated_at = attr.ib(type=datetime)

    links_to_overwrite = attr.ib(type=LinkCollection)

    @property
    def files(self):
        """Convenience property to mimic Snapshot.files"""
        return self.composed_files()

    def composed_files(self, paths=None):
        """
        Return a dict analogous to Snapshot.files ({path: FileInfo})

        Draft implements this by combining our deltas (`files_to_overwrite`)
        and the `base_snapshot.files`. If a `paths` argument is specified, we
        will only apply those deltas. So if your draft has five files that have
        changed, but you want to only apply two of those files in your view of
        a draft's files, you can pass those two files in a list.
        """
        if paths is not None:
            files_to_overwrite = {path: self.files_to_overwrite[path] for path in paths}
        else:
            files_to_overwrite = self.files_to_overwrite

        # Pylint doesn't understand that StagedDraft.base_snapshot is a Snapshot
        # pylint: disable=no-member
        base_snapshot_files = self.base_snapshot.files if self.base_snapshot else {}
        merged_files = {**base_snapshot_files, **files_to_overwrite}
        return {
            path: file_info
            for path, file_info in merged_files.items()
            if file_info is not None
        }

    def composed_links(self):
        """
        Return a LinkCollection that combines overrides with the base Snapshot.
        """
        if not self.base_snapshot:
            base_links = LinkCollection(self.bundle_uuid, [])
        else:
            base_links = self.base_snapshot.links

        return base_links.with_updated_links(self.links_to_overwrite)

    def is_deleted(self, path):
        return (path in self.files_to_overwrite) and (self.files_to_overwrite[path] is None)

    class NotFoundError(Exception):
        pass


class SnapshotRepo:
    """
    This is the interface for storing the actual files that make up a Bundle.

    The responsibilities of this class are:
    * Create a new Snapshot from a set of files.
    * Generate URLs for resources within a Bundle.
    * Validate Links.

    # Important note about Versions/Snapshots:

    Numbered versions don't exist at this level -- versioning here is entirely
    based on content, so basically some kind of hash. BundleVersion is what
    manages numbered versioning and points to an entry in SnapshotRepo. To
    make that distinction clearer, we'll say that SnapshotRepo creates
    Snapshots, which BundleVersion will point to.

    This also helps us to mitigate the concurrency issue. Creating a snapshot
    might take a long time (say to upload all the assets). If we were assigning
    a version_num at the start of that process, we might run into a race
    condition with a process that starts a few seconds later, because they'll
    both see the database reflecting the same state. But if BundleStore creates
    only hash-versioned Snapshots, then it can do all that work and the only
    time version_num is read and incremented for a new entry is when
    SnapshotRepo emits a signal that BundleVersion catches.

    # Extensibility

    Basic: We use Django's File Storage API, which means that one extension
    point is the Storages layer. This is the layer we'd plug into to start,
    using django-storages for things like S3 and equivalents, and some in-memory
    backing store for running tests.

    Advanced: It's possible that one day we'll want to swap out the entire
    SnapshotRepo to adopt a radically different versioning store like git
    (+LFS). In that case, we'd create a new class that implements the same
    public BundleStore interface. To keep the option open, we'd want to make
    sure that things going in and out of BundleStore don't leak storage
    implementation details.

    # Storage conventions:

    Metadata JSON file describing the contents of a Snapshot and basic metadata:
        {bundle_uuid}/snapshots/{snapshot_digest.hex()}.json

    Data for each file:
        {bundle_uuid}/snapshot_data/{file_hash.hexdigest()}

    Notes:
    * To make S3 partitioning work correctly, we need to lead with the UUID or
      some other randomly distributed string.
    * A data file is stored with a hash as a name since it could be referenced
      by different paths in different versions.
    """
    def __init__(self, storage=None):
        self.storage = storage or default_storage

    def get(self, bundle_uuid: UUID, snapshot_digest: bytes) -> Snapshot:
        """
        Return a snapshot.

        TODO: There's no meaningful error handling yet.
        """
        storage_path = self._summary_path(bundle_uuid, snapshot_digest)
        try:
            with self.storage.open(storage_path) as snapshot_file:
                snapshot_json = json.load(snapshot_file)
        except FileNotFoundError:
            logger.error(
                "Snapshot %s,%s not found at: %s",
                bundle_uuid,
                snapshot_digest.hex(),
                storage_path,
            )
            raise Snapshot.NotFoundError(
                f"Snapshot {snapshot_digest.hex()} for Bundle {bundle_uuid} not found"
            )

        print(snapshot_json['links'])

        return Snapshot(
            bundle_uuid=bundle_uuid,
            files=FileInfo.from_json_dict(snapshot_json['files']),
            links=LinkCollection.from_json_dict(
                bundle_uuid, snapshot_json.get('links', {})
            ),
            hash_digest=bytes_from_hex_str(snapshot_json['hash_digest']),
            created_at=parse_utc_iso8601_datetime(snapshot_json['created_at']),
        )

    def create(self, bundle_uuid, paths_to_files, links=None):
        """
        Save the files, create a Snapshot object and save its JSON serialization to storage.
        """
        if links is None:
            links = LinkCollection(bundle_uuid, [])

        files = {}
        for path, file_obj in paths_to_files.items():
            files[str(path)] = self._save_file(bundle_uuid, path, file_obj)

        return self._create(bundle_uuid, files, links)

    def url(self, snapshot, path):
        """Return a user-accessible URL to download a path from this Snapshot."""
        file_info = snapshot.files[path]
        storage_path = self._file_data_path(snapshot.bundle_uuid, file_info.hash_digest.hex())
        return self.storage.url(storage_path)

    def open(self, snapshot, path):
        file_info = snapshot.files[path]
        storage_path = self._file_data_path(snapshot.bundle_uuid, file_info.hash_digest.hex())
        return self.storage.open(storage_path, 'rb')

    @classmethod
    def _summary_path(cls, bundle_uuid, snapshot_digest):
        return f'{bundle_uuid}/snapshots/{snapshot_digest.hex()}.json'

    @classmethod
    def _file_data_path(cls, bundle_uuid, file_hash):
        return f'{bundle_uuid}/snapshot_data/{file_hash}'

    def _save_file(self, bundle_uuid, path, data, public=False):
        """
        Save file at path and return a FileInfo object for it.
        """
        file_hash = FileInfo.generate_hash(data)
        data_write_location = self._file_data_path(bundle_uuid, file_hash.hexdigest())
        if not self.storage.exists(data_write_location):
            self.storage.save(data_write_location, data)
        return FileInfo(
            path=path, public=public, size=data.size, hash_digest=file_hash.digest()
        )

    def _create(self, bundle_uuid, files, links):
        """
        Create a Snapshot object and save its JSON serialization to storage.
        """
        print(f"LINKS! {links.names_to_links}")

        snapshot = Snapshot.create(bundle_uuid=bundle_uuid, files=files, links=links)
        summary_json_str = json.dumps(snapshot, cls=BundleDataJSONEncoder, indent=2, sort_keys=True)
        summary_path = self._summary_path(bundle_uuid, snapshot.hash_digest)

        if not self.storage.exists(summary_path):
            self.storage.save(summary_path, ContentFile(summary_json_str))

        snapshot_created.send(
            SnapshotRepo,
            bundle_uuid=bundle_uuid,
            hash_digest=snapshot.hash_digest,
        )
        logger.info(
            "Created Snapshot %s for Bundle %s", snapshot.hash_digest, bundle_uuid
        )

        return snapshot


class DraftRepo:
    """
    Similar to SnapshotRepo, except that this class stores mutable Drafts.

    DraftRepos know about SnapshotRepos, but not the other way around.
    """
    class SaveError(Exception):
        pass

    def __init__(self, snapshot_repo, storage=None):
        """
        A Draft is basically a diff over a Snapshot, so we need a pointer back
        to where we can get Snapshot data.
        """
        self.snapshot_repo = snapshot_repo
        self.storage = storage or default_storage

    @classmethod
    def _data_file_path(cls, draft_uuid, file_path):
        """Path to a file that we've edited in our Draft."""
        return f'{draft_uuid}/data/{file_path}'

    @classmethod
    def _summary_path(cls, draft_uuid):
        return f'{draft_uuid}/summary.json'

    def _overwrite(self, path, file_obj):
        # There's gotta be a better way, but for now...
        if self.storage.exists(path):
            self.storage.delete(path)
        return self.storage.save(path, file_obj)

    def _save_summary_file(self, draft):
        summary_path = self._summary_path(draft.uuid)
        draft_summary_json = self.serialized_draft_summary(draft)
        self._overwrite(summary_path, ContentFile(draft_summary_json))

    @classmethod
    def serialized_draft_summary(cls, draft):
        return json.dumps(draft, cls=BundleDataJSONEncoder, indent=2, sort_keys=True)

    def get(self, draft_uuid: UUID) -> StagedDraft:
        """
        Get a StagedDraft object by UUID.

        Raises a StagedDraft.NotFoundError if the UUID does not exist.
        """
        summary_path = self._summary_path(draft_uuid)
        if not self.storage.exists(summary_path):
            raise StagedDraft.NotFoundError(f"Draft {draft_uuid} not found in {self}")

        with self.storage.open(summary_path) as draft_summary_file:
            draft_summary_json = json.load(draft_summary_file)

        bundle_uuid = UUID(draft_summary_json['bundle_uuid'])
        if draft_summary_json['base_snapshot'] is None:
            base_snapshot = None
        else:
            base_snapshot_digest = bytes_from_hex_str(draft_summary_json['base_snapshot'])
            base_snapshot = self.snapshot_repo.get(bundle_uuid, base_snapshot_digest)

        # Assemble the LinkCollection information from the Draft Summary JSON
        links_to_overwrite = LinkCollection(
            bundle_uuid,
            [
                Link(
                    name=name,
                    direct_dependency=Dependency.from_json_dict(
                        link_info["direct"]
                    ),
                    indirect_dependencies=[
                        Dependency.from_json_dict(indirect_dep_info)
                        for indirect_dep_info in link_info["indirect"]
                    ]
                )
                for name, link_info
                in draft_summary_json.get("links_to_overwrite", {}).items()
            ]
        )

        return StagedDraft(
            uuid=UUID(draft_summary_json['uuid']),
            bundle_uuid=bundle_uuid,
            name=draft_summary_json['name'],
            base_snapshot=base_snapshot,
            files_to_overwrite=FileInfo.from_json_dict(
                draft_summary_json['files_to_overwrite']
            ),
            links_to_overwrite=links_to_overwrite,
            created_at=parse_utc_iso8601_datetime(draft_summary_json['created_at']),
            updated_at=parse_utc_iso8601_datetime(draft_summary_json['updated_at']),
        )

    def create(self, draft_uuid: UUID, bundle_uuid: UUID, name: str,
               base_snapshot: Snapshot, created_at=None) -> StagedDraft:
        """Create and return a StagedDraft based on a Snapshot."""
        created_at = created_at or datetime.now(timezone.utc)
        draft = StagedDraft(
            uuid=draft_uuid,
            bundle_uuid=bundle_uuid,
            name=name,
            base_snapshot=base_snapshot,
            files_to_overwrite={},
            links_to_overwrite=LinkCollection(bundle_uuid, []),
            created_at=created_at,
            updated_at=created_at,
        )
        self._save_summary_file(draft)
        return draft

    def open(self, draft, file_path, mode='rb'):
        """
        Open a file from a Draft.

        We might get rid of this method altogether in order to prevent us from
        shooting ourselves in the foot performance-wise. For now it's necessary
        for Snapshot creation because that's a copy, as opposed to a move
        operation.
        """
        # TODO: Do an explicit check here to make sure we're not mixing snapshots
        # and Drafts

        if draft.is_deleted(file_path):
            # TODO: throw exception here instead?
            return None
        if file_path in draft.files_to_overwrite:
            return self.storage.open(self._data_file_path(draft.uuid, file_path), mode)
        return self.snapshot_repo.open(draft.base_snapshot, file_path)

    def delete(self, draft_uuid):
        """
        Delete the StagedDraft.

        We don't remove the directory for the draft entirely because that's not
        supported in all storage backends.
        """
        staged_draft_to_delete = self.get(draft_uuid)
        for file_path in staged_draft_to_delete.files_to_overwrite:
            storage_path = self._data_file_path(draft_uuid, file_path)
            self.storage.delete(storage_path)
        self.storage.delete(self._summary_path(draft_uuid))

    def commit(self, draft, paths=None, committed_at=None):
        """
        Commit the Draft as a Snapshot.

        If `paths` is provided, we will commit only those files.
        """
        with self.file_mapping(draft, paths) as draft_files:
            new_snapshot = self.snapshot_repo.create(
                draft.bundle_uuid,
                draft_files,
                draft.composed_links(),
            )

        # Update the Draft to reflect the committed changes, keeping in mind
        # that we might have only committed a subset of files.
        committed_at = committed_at or datetime.now(timezone.utc)

        # This also handles the "deleted" file case, since a delete has a
        # file_info of None and that will match with getting a None from
        # new_snapshot.files.get(path).
        files_to_overwrite = {
            path: file_info
            for path, file_info in draft.files_to_overwrite.items()
            if file_info != new_snapshot.files.get(path)
        }

        updated_draft = attr.evolve(
            draft,
            base_snapshot=new_snapshot,
            files_to_overwrite=files_to_overwrite,
            updated_at=committed_at,
        )
        old_draft_paths_to_delete = [
            path for path in draft.files_to_overwrite
            if path not in updated_draft.files_to_overwrite or updated_draft.is_deleted(path)
        ]

        # Actual storages file writes...
        for path in old_draft_paths_to_delete:
            self.storage.delete(self._data_file_path(draft.uuid, path))
        self._save_summary_file(updated_draft)

        return (new_snapshot, updated_draft)

    @contextmanager
    def file_mapping(self, draft, paths=None):
        """
        Given a Draft, return a mapping of paths -> File objects (not FileInfo)

        This is useful if you want the current Draft state of all Files so that
        you can create a Snapshot from it.

        If you supply an iterable of paths, it means you want the file_mapping
        to only reflect the delta of this Draft over the base_snapshot for those
        particular files. This is useful if you want to commit the changes from
        a specific set of files instead of the entire Draft.
        """
        # This is a dict of {path: FileInfo} with all files in the Bundle, but
        # with the Draft versions of those files only for `paths` if specified.
        # (This is to support partial commits.)
        composed_files = draft.composed_files(paths)

        def file_for_path(path):
            # If it hasn't changed from the snapshot, serve it from the snapshot
            if draft.base_snapshot and composed_files[path] == draft.base_snapshot.files.get(path):
                return self.snapshot_repo.open(draft.base_snapshot, path)
            # Otherwise, look for it in our draft files.
            return self.open(draft, path)

        paths_to_files = {path: file_for_path(path) for path in composed_files}
        try:
            yield paths_to_files
        finally:
            for open_file in paths_to_files.values():
                open_file.close()

    def update(self, draft_uuid, files, dependencies=None, updated_at=None) -> StagedDraft:
        """
        Update the draft.

        `files` is a dict with paths for keys and Files for values.

        `dependencies` is a mapping of Link names to direct Dependencies. A full
        Link is name + direct dependencies + indirect dependencies. So we have
        to verify that the requested direct dependency actually exists and then
        retrieve the indirect dependencies.

        This method has to figure out what the indirect dependencies are.
        """
        if dependencies is None:
            dependencies = {}

        existing_draft = self.get(draft_uuid)
        new_files = self._new_files_for_update(existing_draft, files)
        new_links_col = self._new_links_for_update(existing_draft, dependencies)

        new_draft = attr.evolve(
            existing_draft,
            files_to_overwrite={**existing_draft.files_to_overwrite, **new_files},
            links_to_overwrite=new_links_col,
            updated_at=updated_at or datetime.now(timezone.utc)
        )

        self._save_summary_file(new_draft)

        return new_draft

    def _new_links_for_update(self, existing_draft, dependencies):
        # First, let's only find the dependencies we actually need to update...
        dependency_updates = {
            name: dep
            for name, dep in dependencies.items()
            if dep != existing_draft.links_to_overwrite.get_direct_dep(name)
        }

        # Now we find the indirect dependencies for all of these
        new_links = []
        for name, dep in dependency_updates.items():
            indirects = [] if dep is None else self._get_indirects(dep)
            if dep is not None:
                new_links.append(Link(name, dep, indirects))

        return LinkCollection(existing_draft.bundle_uuid, new_links)

    def _get_indirects(self, dep):
        """
        Given a direct dependency, find and return all indirect dependencies.

        This requires accessing the Snapshot for the dependency.
        """
        dep_snapshot = self.snapshot_repo.get(dep.bundle_uuid, dep.snapshot_digest)
        return dep_snapshot.links.all_dependencies()

    def _new_files_for_update(self, existing_draft, files):
        # First write all the files out in their appropriate draft space.
        new_files_written = {}
        for path, django_file in files.items():
            if not is_safe_file_path(path):
                raise DraftRepo.SaveError(f'"{path}" is not a valid file name')
            storage_path = self._data_file_path(existing_draft.uuid, path)

            # If the django_file is None, it means we want a delete
            if django_file is None:
                file_info = None
                if self.storage.exists(storage_path):
                    self.storage.delete(storage_path)
            else:
                file_info = FileInfo(
                    path=path,
                    public=False,  # Hardcoded for now -- maybe get rid of this from FileInfo altogether?
                    size=django_file.size,
                    hash_digest=FileInfo.generate_hash(django_file).digest()
                )
                self._overwrite(storage_path, django_file)

            new_files_written[path] = file_info
        return new_files_written

    def url(self, draft, path):
        """Return URL of Draft file, fallback to Snapshot if unmodified."""
        if draft.is_deleted(path):
            return None

        if path in draft.files_to_overwrite:
            data_file_path = self._data_file_path(draft.uuid, path)
            return self.storage.url(data_file_path)

        return self.snapshot_repo.url(draft.base_snapshot, path)


class BundleDataJSONEncoder(json.JSONEncoder):
    """Default JSON serialization."""
    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, FileInfo):
            return [o.public, o.size, o.hash_digest.hex()]
        elif isinstance(o, UUID):
            return str(o)
        elif isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, Snapshot):
            return {
                'bundle_uuid': o.bundle_uuid,
                'hash_digest': o.hash_digest.hex(),
                'files': o.files,
                'links': o.links,
                'created_at': o.created_at,
                '_type': 'snapshot',
                '_version': 1,
            }
        elif isinstance(o, StagedDraft):
            if o.base_snapshot is None:
                base_snapshot = None
            else:
                base_snapshot = o.base_snapshot.hash_digest.hex()
            return {
                'uuid': o.uuid,
                'bundle_uuid': o.bundle_uuid,
                'name': o.name,
                'base_snapshot': base_snapshot,
                'files_to_overwrite': o.files_to_overwrite,
                'links_to_overwrite': o.links_to_overwrite,
                'created_at': o.created_at,
                'updated_at': o.updated_at,
                '_type': 'draft',
                '_version': 1,
            }
        elif isinstance(o, LinkCollection):
            return {
                link.name: {
                    "direct": link.direct_dependency,
                    "indirect": link.indirect_dependencies
                }
                for link in o
            }
        # Links are rarely serialized outside a LinkCollection
        elif isinstance(o, Link):
            return {
                "name": o.name,
                "direct": o.direct_dependency,
                "indirect": o.indirect_dependencies,
            }
        elif isinstance(o, Dependency):
            return {
                "bundle_uuid": o.bundle_uuid,
                "version": o.version,
                "snapshot_digest": o.snapshot_digest.hex(),
            }

        return json.JSONEncoder.default(self, o)


@contextmanager
def files_from_disk(bundle_data_path):
    """
    Given a pathlib.Path, return file mapping suitable for Snapshot.

    Returned dictionary will map from str to django.core.files.File objects.

    This will recursively scan subdirectories.
    """
    paths_to_files = {
        path.relative_to(*path.parts[:1]): File(open(path, "rb"))
        for path in bundle_data_path.rglob('*')
        if path.is_file()
    }
    try:
        yield paths_to_files
    finally:
        for open_file in paths_to_files.values():
            open_file.close()


def is_safe_file_path(path):
    """
    Is this a safe file path for a snapshot/draft file name?

    Sub-dirs are allowed, but no crawling up to parent paths.
    """
    if '//' in path:  # This will parse away, but probably indicates a bug...
        return False

    path_obj = Path(path)
    if Path.is_absolute(path_obj):
        return False
    if '..' in path_obj.parts:
        return False

    # Fine tune this later...
    if len(path) > 500:
        return False
    return True


def create_hash(start_data=b''):
    """Whenever there's a need for hashing, we use 20-byte BLAKE2b."""
    return hashlib.blake2b(start_data, digest_size=20)  # pylint: disable=no-member


def parse_utc_iso8601_datetime(datetime_str):
    """
    Create a UTC datetime from a str generated by calling datetime.isoformat()

    Python's datetime module is incapable of parsing the ISO 8601 string output
    that it itself produces (it can't do timezones with ":" in them), so we need
    to only parse the part before that and then manually add UTC timezone
    information into it. This is a hack -- it will ignore whatever timezone is
    really there and force it to be interpreted as UTC.

    This hack should not be necessary once we move to Python 3.7 and can be
    replaced with datetime.fromisoformat().
    """
    parsed_dt = datetime.strptime(datetime_str, r'%Y-%m-%dT%H:%M:%S.%f+00:00')
    return parsed_dt.astimezone(timezone.utc)


def bytes_from_hex_str(hex_str):
    """Return bytes given a hexidecimal string representation of binary data."""
    if hex_str is None:
        return None
    return codecs.decode(hex_str, 'hex')
