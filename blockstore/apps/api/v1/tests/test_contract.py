"""
Tests for v1 REST API
=====================

These tests help guarantee API functionality and backwards compatibility. A
test failure in this module should represent an instance where we're making a
breaking change to the REST API contract. Internal refactoring should never
result in breaking these tests. To that end:

1. Data setup should happen via REST API. No factories or models.
2. Introspection should happen via REST API. No query counts or model queries.
3. Tests should look for particular fields of interest but not assume they know
   every field, since additional ones can be added in a backwards compatible
   way. Testing one attribute at a time also makes it a lot easier to see when
   there is a regression, instead of trying to look at a large diff.

This file has many 😀 emojis in strings to test for Unicode encoding/decoding
related issues.
"""
import re

from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from blockstore.apps.bundles.tests.storage_utils import isolate_test_storage
from blockstore.apps.api.constants import UUID4_REGEX
from .helpers import (
    create_bundle_with_history, encode_str_for_draft, response_str_file, response_data
)


@isolate_test_storage
class CollectionsTestCase(TestCase):
    """Test basic Collections CRUD operations."""

    def setUp(self):
        super().setUp()
        self.client = APIClient()

    def test_list_of_one(self):
        """
        Make sure list[0] and detail have the same representation.
        """
        create_response = self.client.post(
            '/api/v1/collections',
            data={'title': "First Collection 😀"},
        )
        assert create_response.status_code == status.HTTP_201_CREATED

        list_response = self.client.get('/api/v1/collections')
        assert list_response.status_code == status.HTTP_200_OK
        list_data = response_data(list_response)
        assert len(list_data) == 1

        first_col_data = list_data[0]
        assert first_col_data['title'] == "First Collection 😀"

        detail_response = self.client.get(first_col_data['url'])
        assert detail_response.status_code == status.HTTP_200_OK
        detail_data = response_data(detail_response)

        # Detail data may eventually have more than what's shown in list, but
        # any key that shows up in list should show up in detail as well.
        for key in first_col_data:
            assert first_col_data[key] == detail_data[key]

    def test_list_multiple(self):
        """
        Create second Collection and make sure it's represented.

        We make no ordering guarantees of the default Collections endpoint.
        """
        create_1_response = self.client.post(
            '/api/v1/collections',
            data={'title': "First Collection 😀"},
        )
        create_1_data = response_data(create_1_response)
        create_2_response = self.client.post(
            '/api/v1/collections',
            data={'title': 'Second 😀 Collection!'},
        )
        create_2_data = response_data(create_2_response)

        list_response = self.client.get('/api/v1/collections')
        assert list_response.status_code == status.HTTP_200_OK
        list_data = response_data(list_response)
        assert len(list_data) == 2

        # Ordering not guarnateed, but both should be present in list.
        assert create_1_data in list_data
        assert create_2_data in list_data
        assert create_1_data != create_2_data

    def test_create(self):
        """Create a Collection and test the returned output."""
        create_response = self.client.post(
            '/api/v1/collections',
            data={'title': '😀 Create Test!'},
        )
        assert create_response.status_code == status.HTTP_201_CREATED
        create_data = response_data(create_response)
        assert create_data['title'] == '😀 Create Test!'
        assert re.match(UUID4_REGEX, create_data['uuid'])

        detail_response = self.client.get(create_data['url'])
        assert detail_response.status_code == status.HTTP_200_OK
        detail_data = response_data(detail_response)
        assert create_data == detail_data

    def test_update(self):
        create_response = self.client.post(
            '/api/v1/collections',
            data={'title': '😀 Update Test!'},
        )
        create_data = response_data(create_response)
        update_response = self.client.put(
            create_data['url'],
            data={'title': 'New 😀 Title!'}
        )
        assert update_response.status_code == status.HTTP_200_OK
        update_data = response_data(update_response)

        assert update_data['url'] == create_data['url']
        assert update_data['uuid'] == create_data['uuid']
        assert update_data['title'] == 'New 😀 Title!'

        # Try re-fetching it after it's been modified.
        detail_response = self.client.get(create_data['url'])
        detail_data = response_data(detail_response)
        assert update_data == detail_data


@isolate_test_storage
class BundlesMetadataTestCase(TestCase):
    """Test basic Bundles Metadata (not content)"""

    def setUp(self):
        super().setUp()
        self.client = APIClient()

        collection_response = self.client.post(
            '/api/v1/collections',
            data={'title': 'Bundle Default Collection 😀'}
        )
        collection_data = response_data(collection_response)
        self.collection_uuid_str = collection_data['uuid']

    def test_create(self):
        create_response = self.client.post(
            '/api/v1/bundles',
            data={
                'collection_uuid': self.collection_uuid_str,
                'description': "This is a 😀😀😀😀 Bundle",
                'slug': 'happy',
                'title': "Happy Bundle 😀"
            }
        )
        assert create_response.status_code == status.HTTP_201_CREATED
        create_data = response_data(create_response)
        assert create_data['collection'] == f'http://testserver/api/v1/collections/{self.collection_uuid_str}'
        assert create_data['collection_uuid'] == self.collection_uuid_str
        assert create_data['description'] == "This is a 😀😀😀😀 Bundle"
        assert create_data['drafts'] == {}
        assert create_data['slug'] == 'happy'
        assert create_data['title'] == "Happy Bundle 😀"
        assert re.match(UUID4_REGEX, create_data['uuid'])
        assert create_data['url'] == f"http://testserver/api/v1/bundles/{create_data['uuid']}"
        assert create_data['versions'] == []

        # Check that the GET returns the same thing
        detail_response = self.client.get(create_data['url'])
        assert detail_response.status_code == status.HTTP_200_OK
        detail_data = response_data(detail_response)
        assert detail_data == create_data

    def test_list(self):
        for i in range(10):
            self.client.post(
                '/api/v1/bundles',
                data={
                    'collection_uuid': self.collection_uuid_str,
                    'description': f"Happy Bundle {i} 😀 is Happy!",
                    'slug': f'happy_{i}',
                    'title': f"Happy Bundle {i} 😀"
                }
            )
        list_response = self.client.get('/api/v1/bundles')
        assert list_response.status_code == status.HTTP_200_OK
        list_data = response_data(list_response)
        assert len(list_data) == 10


@isolate_test_storage
class DraftsTest(TestCase):
    """Test creation, editing, and commits of content to Bundles."""

    def setUp(self):
        super().setUp()
        self.client = APIClient()

        collection_response = self.client.post(
            '/api/v1/collections',
            data={'title': 'Bundle Default Collection 😀'}
        )
        collection_data = response_data(collection_response)

        bundle_response = self.client.post(
            '/api/v1/bundles',
            data={
                'collection_uuid': collection_data['uuid'],
                'description': "Draft Test 😀😀😀😀 Bundle",
                'slug': 'draft_test',
                'title': "Draft Test Bundle 😀"
            }
        )
        self.bundle_data = response_data(bundle_response)

    def test_basic_draft_commit(self):
        """Happy path test of Draft commits."""
        create_draft_response = self.client.post(
            '/api/v1/drafts',
            {
                'bundle_uuid': self.bundle_data['uuid'],
                'name': 'studio_draft',
                'title': "For DraftsTest.test_basic_draft_commit",
            }
        )
        assert create_draft_response.status_code == status.HTTP_201_CREATED
        draft_data = response_data(create_draft_response)
        assert re.match(UUID4_REGEX, draft_data['uuid'])
        assert draft_data['url'] == f'http://testserver/api/v1/drafts/{draft_data["uuid"]}'
        assert draft_data['bundle_uuid'] == self.bundle_data['uuid']
        assert draft_data['bundle'] == self.bundle_data['url']
        assert draft_data['name'] == 'studio_draft'
        assert 'staged_draft' not in draft_data

        # Getting the result from a retrieve (GET on individual Draft) should
        # yield the same data, except that it should *also* contain the
        # metadata about the files inside (staged_draft)
        detail_response = self.client.get(draft_data['url'])
        detail_data = response_data(detail_response)
        for key in draft_data:
            assert detail_data[key] == draft_data[key]

        staged_draft_data = detail_data['staged_draft']
        assert staged_draft_data['base_snapshot'] is None  # No commit yet.

        # Bundle should show our draft, but no versions (no commits yet)
        bundle_detail_response = self.client.get(f'/api/v1/bundles/{draft_data["bundle_uuid"]}')
        bundle_detail_data = response_data(bundle_detail_response)
        assert bundle_detail_data['drafts'] == {'studio_draft': draft_data['url']}
        assert bundle_detail_data['versions'] == []

        # Add a simple file
        draft_patch_response = self.client.patch(
            draft_data['url'],
            data={
                'files': {
                    'hello.txt': encode_str_for_draft("Hello World! 😀")
                }
            },
            format='json'
        )
        assert draft_patch_response.status_code == status.HTTP_204_NO_CONTENT

        # Now commit it
        commit_response = self.client.post(f'/api/v1/drafts/{draft_data["uuid"]}/commit')
        assert commit_response.status_code == status.HTTP_201_CREATED

        # Now grab the Bundle again and check that a new version exists...
        bundle_detail_response = self.client.get(f'/api/v1/bundles/{draft_data["bundle_uuid"]}')
        bundle_detail_data = response_data(bundle_detail_response)
        assert len(bundle_detail_data['versions']) == 1

        # Now get the file we committed
        bundle_version_detail_data = response_data(
            self.client.get(bundle_detail_data['versions'][0])
        )
        file_url = bundle_version_detail_data['snapshot']['files']['hello.txt']['url']
        file_response = self.client.get(file_url)
        assert response_str_file(file_response) == "Hello World! 😀"

    def test_editing_errors(self):
        create_response = self.client.post(
            '/api/v1/drafts',
            {
                'bundle_uuid': self.bundle_data['uuid'],
                'name': 'studio_draft',
                'title': "For DraftsTest.test_editing_errors 😀",
            }
        )
        create_data = response_data(create_response)
        draft_url = create_data['url']

        # This won't work because the input is not base64 encoded.
        patch_response = self.client.patch(
            draft_url, data={'files': {'hello.txt': b"I'm Not Base64!"}}, format='json',
        )
        assert patch_response.status_code == status.HTTP_400_BAD_REQUEST

        # This won't work because the filename can't have .. in the path.
        patch_response = self.client.patch(
            draft_url, data={'files': {'../hello.txt': b""}}, format='json',
        )
        assert patch_response.status_code == status.HTTP_400_BAD_REQUEST

    def test_editing_draft(self):
        create_response = self.client.post(
            '/api/v1/drafts',
            {
                'bundle_uuid': self.bundle_data['uuid'],
                'name': 'studio_draft',
                'title': "For DraftsTest.test_editing_draft 😀",
            }
        )
        create_data = response_data(create_response)
        draft_url = create_data['url']

        # Create some files and commit...
        self.client.patch(
            draft_url,
            data={
                'files': {
                    'empty.txt': encode_str_for_draft(""),
                    'hawaii.txt': encode_str_for_draft("Aloha!"),
                    'korea.txt': encode_str_for_draft("안녕하세요!"),
                }
            },
            format='json',
        )
        commit_data = response_data(self.client.post(f'{draft_url}/commit'))
        bundle_version_url = commit_data['bundle_version']
        bundle_version_data = response_data(self.client.get(bundle_version_url))
        snapshot_files = bundle_version_data['snapshot']['files']
        updated_draft_data = response_data(self.client.get(draft_url))
        draft_files = updated_draft_data['staged_draft']['files']

        assert updated_draft_data['staged_draft']['base_snapshot'] == bundle_version_data['snapshot']['hash_digest']
        for file_name in draft_files:
            assert draft_files[file_name]['url'] == snapshot_files[file_name]['url']
            assert draft_files[file_name]['size'] == snapshot_files[file_name]['size']
            assert draft_files[file_name]['hash_digest'] == snapshot_files[file_name]['hash_digest']
            assert draft_files[file_name]['modified'] is False

        # Now update some files...
        self.client.patch(
            draft_url,
            data={
                'files': {
                    'empty.txt': None,  # Delete this file
                    'hawaii.txt': encode_str_for_draft("Aloha a hui hou!"),
                    # korea.txt is not mentioned, so it will be untouched
                }
            },
            format='json',
        )
        updated_draft_data = response_data(self.client.get(draft_url))
        draft_files = updated_draft_data['staged_draft']['files']
        assert 'empty.txt' not in draft_files
        assert draft_files['hawaii.txt']['modified'] is True
        assert draft_files['korea.txt']['modified'] is False

        # Now commit...
        commit_data = response_data(self.client.post(f'{draft_url}/commit'))
        bundle_version_url = commit_data['bundle_version']
        bundle_version_data = response_data(self.client.get(bundle_version_url))
        snapshot_files = bundle_version_data['snapshot']['files']
        assert 'empty.txt' not in snapshot_files  # Was deleted
        hawaii_text = response_str_file(
            self.client.get(snapshot_files['hawaii.txt']['url'])
        )
        assert hawaii_text == "Aloha a hui hou!"
        korea_text = response_str_file(
            self.client.get(snapshot_files['korea.txt']['url'])
        )
        assert korea_text == "안녕하세요!"


@isolate_test_storage
class LinksTest(TestCase):
    """Test creating and following Links."""

    def setUp(self):
        """
        We need to build up a little history to make the Links tests meaningful.

        Here, we're going to set up a Course Bundle that Links to a Library
        Bundle in a different Collection.
        """
        super().setUp()
        self.client = APIClient()

        self.library_collection_data = response_data(
            self.client.post(
                '/api/v1/collections', data={'title': 'Links Library Collection'}
            )
        )
        self.course_collection_data = response_data(
            self.client.post(
                '/api/v1/collections', data={'title': 'Links Course Collection'}
            )
        )

        self.library_bundle_data = create_bundle_with_history(
            self.client,
            self.library_collection_data['uuid'],
            "Dogs Library Bundle 🐶",
            [
                {'dog.txt': encode_str_for_draft("Rusty! 🐕")},
                {'dog.txt': encode_str_for_draft("Jack! 🐕")},
                {'dog.txt': encode_str_for_draft("Clyde! 🐕")},
            ]
        )
        self.course_bundle_data = create_bundle_with_history(
            self.client,
            self.course_collection_data['uuid'],
            "Course Bundle",
            [
                {
                    'overview.md': encode_str_for_draft(
                        "## TODO: 🤫 Make this an RST file instead!"
                    )
                }
            ]
        )

    def test_simple_links(self):
        """
        Links: Create (in Draft), Commit, Delete
        """
        course_draft_url = self.course_bundle_data['drafts']['test_draft']
        new_link_data = {
            "dog_library": {
                "bundle_uuid": self.library_bundle_data['uuid'],
                "version": 3  # Use the latest -- TODO: should we default to latest?
            }
        }
        self.client.patch(
            course_draft_url, data={'links': new_link_data}, format='json',
        )
        draft_data = response_data(self.client.get(course_draft_url))
        draft_link_data = draft_data['staged_draft']['links']['dog_library']

        assert draft_link_data['modified'] is True
        assert draft_link_data['direct']['bundle_uuid'] == self.library_bundle_data['uuid']
        assert draft_link_data['direct']['version'] == 3
        assert draft_link_data['direct'].get('snapshot_digest') is not None
        assert draft_link_data['indirect'] == []

        commit_resp_data = response_data(
            self.client.post(course_draft_url + '/commit')
        )
        new_bv_url = commit_resp_data['bundle_version']
        new_bv_data = response_data(self.client.get(new_bv_url))
        bv_link_data = new_bv_data['snapshot']['links']['dog_library']

        # The link data should be identical except that only Drafts have a
        # 'modified' key...
        draft_link_data.pop('modified')
        assert draft_link_data == bv_link_data

        draft_data = response_data(
            self.client.patch(
                course_draft_url, data={'links': {'dog_library': None}}, format='json'
            )
        )
        print(draft_data)
        assert False
