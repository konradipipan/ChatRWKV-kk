
def produce_dd(phones_data: list[dict]) ->list[dict]:

    # Prepare dict dataset
    dataset_dicts = []
    for phd in phones_data:
        phd['nazwasrodka'] = phd['nazwasrodka'].replace('\/', '').replace('\\/', '').replace('/', '')
        tokens = phd['nazwasrodka'].split(' ')
        new_dict = {'nazwasrodka' : phd['nazwasrodka'],
                    'taxonomy' : ''}
        if phd['T0'] and phd['T0'] in tokens:
            new_dict['taxonomy'] += phd['T0']
            new_dict['taxonomy'] += ' '

        if phd['T1'] and phd['T1'] in tokens:
            new_dict['taxonomy'] += phd['T1']
            new_dict['taxonomy'] += ' '
        for i in [2, 3, 4]:
            if phd[f'T{i}'] and phd[f'T{i}'] in tokens:
                new_dict['taxonomy'] += phd[f'T{i}']
                new_dict['taxonomy'] += ' '
        new_dict['taxonomy'] = new_dict['taxonomy'].rstrip(' ')
        dataset_dicts.append(new_dict)
    return dataset_dicts