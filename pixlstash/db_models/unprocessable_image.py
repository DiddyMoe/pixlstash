"""The durable copy of :class:`~pixlstash.utils.unprocessable_image_registry.UnprocessableImageRegistry`.

One row per picture the background pipeline holds back: an undecodable file,
pinned to the ``(mtime_ns, size)`` it had when it failed; a file whose
location is not mounted; or a file that is there but refuses to be read (no
permission, an I/O error). The last two have no signature: they are held while
the condition lasts, and lift the moment it ends. The registry works from
memory and writes each change through to this table, and reads it back on its
first use after a start, so a restart no longer retries every such picture
once more. The row goes when the registry lifts or forgets the entry, and with
the picture itself (``ON DELETE CASCADE``).
"""

from typing import Optional

from sqlmodel import Column, Field, ForeignKey, Integer, SQLModel

#: ``kind`` of an entry pinned to the file's signature.
KIND_UNDECODABLE = "undecodable"
#: ``kind`` of an entry held while the file's location is missing.
KIND_UNREACHABLE = "unreachable"
#: ``kind`` of an entry held while the file exists but cannot be read.
KIND_UNREADABLE = "unreadable"


class UnprocessableImage(SQLModel, table=True):
    """One picture the work finders leave out, and why."""

    __tablename__ = "unprocessableimage"

    picture_id: int = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("picture.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    kind: str = Field(nullable=False)
    file_path: str = Field(nullable=False)
    # NULL for an unreachable or unreadable entry: it is held while a
    # condition lasts, not pinned to one version of the file.
    mtime_ns: Optional[int] = Field(default=None, nullable=True)
    size: Optional[int] = Field(default=None, nullable=True)
