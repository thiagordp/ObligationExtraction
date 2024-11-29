from src.extraction_validation import calculate_validation_metrics, calculate_statistics, extract_all_as_dict, \
    export_beneficiaries_as_list

def extract_info_as_list(target_info):
    data = extract_all_as_dict()

    all_beneficiaries = []
    for extraction in data:
        beneficiary_str = export_beneficiaries_as_list(extraction, target_info)
        all_beneficiaries.append(beneficiary_str)

    separation_string = "=" * 50
    final_string = f"\n{separation_string}\n".join(all_beneficiaries)

    with open(f"data/validation/list_outputs/{target_info}.txt", "w+") as fp:
        fp.write(final_string)

def main():
    # calculate_validation_metrics()
    # calculate_statistics()
    extract_info_as_list("Beneficiaries")
    extract_info_as_list("Standards")


if __name__ == "__main__":
    main()
