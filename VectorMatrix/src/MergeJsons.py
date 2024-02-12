import json
import avro.schema


SCHEMA1_EXPLCT_INDICATOR = ""
FOUND_DIFFERENCE_INDICATOR = "mismatch_"
ID_FOR_ITEM_IN_LIST = "name"


def is_primitive_type(field):
    return not isinstance(field, list) and not isinstance(field, dict)


def add_duplicate_field(field_name, value1, value2, parent):
    parent[field_name] = value1
    parent[FOUND_DIFFERENCE_INDICATOR + field_name] = value2


def is_all_primitive_types(schema):
    for field in schema.keys():
        if not is_primitive_type(schema[field]):
            return False
    return True


def is_any_dict(list_of_fields):
    for field in list_of_fields:
        if isinstance(field, dict):
            return True
    return False


def union_lists(list1, list2):
    result = list() + list1
    for item in list2:
        if item not in result:
            result.append(item)
    return result


def add_fields_from_first_schema(schema1, schema2, result):
    for field in schema1.keys():
        if field not in schema2:
            result[SCHEMA1_EXPLCT_INDICATOR + field] = schema1[field]
        elif not isinstance(schema1[field], type(schema2[field])):
            # Do nothing here
            pass
        elif isinstance(schema1[field], dict):
            add_fields_from_first_schema(schema1[field], schema2[field], result[field])
        elif isinstance(schema1[field], list):
            if len(schema1[field]) > 0 and is_any_dict(schema1[field]):
                for i in range(len(schema1[field])):
                    # find in list of dictionaries by name

                    if isinstance(schema1[field][i], dict) and ID_FOR_ITEM_IN_LIST in schema1[field][i]:
                        found = False
                        for j in range(len(schema2[field])):
                            if (isinstance(schema2[field][j], dict) and
                                    ID_FOR_ITEM_IN_LIST in schema2[field][j] and
                                    schema1[field][i][ID_FOR_ITEM_IN_LIST] == schema2[field][j][ID_FOR_ITEM_IN_LIST]):
                                add_fields_from_first_schema(schema1[field][i], schema2[field][j], result[field][j])
                                found = True
                                break

                        if found is False:
                            new_item = schema1[field][i]
                            if isinstance(new_item, dict) and ID_FOR_ITEM_IN_LIST in new_item:
                                new_item[ID_FOR_ITEM_IN_LIST] = SCHEMA1_EXPLCT_INDICATOR + new_item[ID_FOR_ITEM_IN_LIST]
                                result[field].append(new_item)
                            else:
                                result[field].append(new_item)
                    elif is_primitive_type(schema1[field][i]):
                        if schema1[field][i] not in schema2[field]:
                            result[field].append(schema1[field][i])
                    else:
                        print('Error: list of dictionaries with no name field and not primitive type')
            elif len(schema1[field]) > 0 and is_primitive_type(schema1[field][0]):
                try:
                    # probably this union is not required. We are anyway doing it from schema2
                    result[field] = union_lists(schema2[field], schema1[field])
                except Exception as e:
                    print('Error', e)
                    result[field] = schema1[field]


def merge_second_schema(schema1, schema2):
    result = {}
    for field in schema2.keys():
        if field in schema1:
            if not isinstance(schema1[field], type(schema2[field])):
                add_duplicate_field(field, schema1[field], schema2[field], result)
            elif is_primitive_type(schema2[field]) and schema1[field] != schema2[field]:
                add_duplicate_field(field, schema1[field], schema2[field], result)
            elif is_primitive_type(schema2[field]) and schema1[field] == schema2[field]:
                result[field] = schema2[field]
            elif isinstance(schema2[field], list):
                list_fields = list()
                if len(schema2[field]) > 0:
                    if is_any_dict(schema2[field]):
                        for j in range(len(schema2[field])):
                            # find in list of dictionaries by name
                            if isinstance(schema2[field][j], dict) and ID_FOR_ITEM_IN_LIST in schema2[field][j]:
                                found = False
                                for k in range(len(schema1[field])):
                                    if (isinstance(schema1[field][k], dict) and
                                            ID_FOR_ITEM_IN_LIST in schema1[field][k] and
                                            schema2[field][j][ID_FOR_ITEM_IN_LIST] == schema1[field][k][
                                                ID_FOR_ITEM_IN_LIST]):
                                        list_fields.append(merge_second_schema(schema1[field][k], schema2[field][j]))
                                        found = True
                                        break
                                if found is False:
                                    list_fields.append(schema2[field][j])
                            elif is_primitive_type(schema2[field][j]):
                                if schema2[field][j] not in list_fields:
                                    list_fields.append(schema2[field][j])
                            else:
                                print('Error: list of dictionaries with no name field and not primitive type')
                    elif is_primitive_type(schema2[field][0]):
                        try:
                            list_fields = union_lists(schema2[field], schema2[field])
                        except Exception as e:
                            print('Error', e)
                            list_fields = schema2[field]
                    else:
                        list_fields = schema2[field]
                result[field] = list_fields
            elif isinstance(schema1[field], dict):
                result[field] = merge_second_schema(schema1[field], schema2[field])
        else:
            result[field] = schema2[field]
    return result


def merge_schemas(schema1, schema2):
    result = merge_second_schema(schema1, schema2)
    # add fields from schema1 that are not in schema2
    add_fields_from_first_schema(schema1, schema2, result)
    return result


def main():

    output_file = '<file path>'
    schema1_file = '<file path>'
    schema2_file = '<file path>'
    schema1 = avro.schema.parse(open(
        '<file path>').read())
    schema2 = avro.schema.parse(open(
        '<file path>').read())

    schema1 = schema1.to_json()
    schema2 = schema2.to_json()

    print('schema1', json.dumps(schema1))
    print('schema2', json.dumps(schema2))
    with open(output_file, 'w') as merged_schema_file:
        json.dump(merge_schemas(schema1, schema2), merged_schema_file, indent=4)

    with open(schema1_file, 'w') as schema1_file:
        json.dump(schema1, schema1_file, indent=4)

    with open(schema2_file, 'w') as schema2_file:
        json.dump(schema2, schema2_file, indent=4)


if __name__ == "__main__":
    main()
