## Sample data exists to show the format of the data, not the data for training.

### You can download the [Patent Dataset](https://drive.google.com/file/d/1So3unr5p_vlYq31gE0Ly07Z2XTvD5QlM/view) used in the paper. Make sure they are under the /data folder.

#### Now, I will explain how to construct the label index. 

```
{"id": "1", "title": ["tokens"], "abstract": ["tokens"], "section": [1, 2], "subsection": [1, 2, 3, 4], "group": [1, 2, 3, 4], "labels": [1, 2, 1+N, 2+N, 3+N, 4+N, 1+N+M, 2+N+M, 3+N+M, 4+N+M]}
```
* "id": just the id.
* "title" & "abstract": it's the word segment (after cleaning stopwords).
* "section": it's the first level category index.
* "subsection": it's the second level category index.
* "group": it's the third level category index.
* "subgroup": it's the fourth level category index.
* "labels": it's the total category which add the index offset.

**Assume that your total category number of level-2 is 100 (*N*=100), of level-3 is 500 (*M*=500). *N* & *M* is the offset for the `labels` attribute.**

the record should be construed as:

```json
{"id": "1", "hashtags": ["token"], "section": [1, 2], "subsection": [1, 2, 3, 4], "group": [1, 2, 3, 4], "labels": [1, 2, 101, 102, 103, 104, 601, 602, 603, 604]}
```
